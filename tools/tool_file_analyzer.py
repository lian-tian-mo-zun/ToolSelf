import json
import os
import base64
import time
import re
from typing import Union, List, Dict, Optional
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.simple_doc_parser import PARAGRAPH_SPLIT_SYMBOL, SimpleDocParser, get_plain_doc
from qwen_agent.tools.storage import KeyNotExistsError, Storage
from qwen_agent.utils.tokenization_qwen import count_tokens, tokenizer
from qwen_agent.utils.utils import get_basename_from_url, hash_sha256
from qwen_agent.settings import DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_WORKSPACE
from pydantic import BaseModel
from openai import OpenAI

TEXT_EXTRACTOR_PROMPT = """Please process the following file content and user goal to extract relevant information:

## **File Content**
{file_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the file content
2. **Key Extraction**: Identify and extract the **most relevant information** from the content, output the **full original context** as far as possible
3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity

**Final Output Format using JSON format with "evidence" and "summary" fields**
"""

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}

TEXT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.txt', '.md', '.html', '.htm', '.csv', '.json', '.xml'}


class Chunk(BaseModel):
    content: str
    metadata: dict
    token: int

    def __init__(self, content: str, metadata: dict, token: int):
        super().__init__(content=content, metadata=metadata, token=token)

    def to_dict(self) -> dict:
        return {'content': self.content, 'metadata': self.metadata, 'token': self.token}


class Record(BaseModel):
    url: str
    raw: List[Chunk]
    title: str

    def __init__(self, url: str, raw: List[Chunk], title: str):
        super().__init__(url=url, raw=raw, title=title)

    def to_dict(self) -> dict:
        return {'url': self.url, 'raw': [x.to_dict() for x in self.raw], 'title': self.title}


class InternalDocParser:

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        self.max_ref_token: int = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)
        self.parser_page_size: int = self.cfg.get('parser_page_size', DEFAULT_PARSER_PAGE_SIZE)

        self.data_root = self.cfg.get('path', os.path.join(DEFAULT_WORKSPACE, 'tools', 'file_analyzer'))
        self.db = Storage({'storage_root_path': self.data_root})

        self.doc_extractor = SimpleDocParser({'structured_doc': True})

    def parse(self, file_path: str, **kwargs) -> dict:
        max_ref_token = kwargs.get('max_ref_token', self.max_ref_token)
        parser_page_size = kwargs.get('parser_page_size', self.parser_page_size)

        url = file_path

        cached_name_chunking = f'{hash_sha256(url)}_{str(parser_page_size)}'
        try:
            record = self.db.get(cached_name_chunking)
            record = json.loads(record)
            print(f'[file_analyzer] Read chunked {url} from cache.')
            return record
        except KeyNotExistsError:
            doc = self.doc_extractor.call({'url': url})

        total_token = 0
        for page in doc:
            for para in page['content']:
                total_token += para['token']

        if doc and 'title' in doc[0]:
            title = doc[0]['title']
        else:
            title = get_basename_from_url(url)

        print(f'[file_analyzer] Start chunking {url} ({title})...')
        time1 = time.time()
        if total_token <= max_ref_token:
            content = [
                Chunk(content=get_plain_doc(doc),
                      metadata={
                          'source': url,
                          'title': title,
                          'chunk_id': 0
                      },
                      token=total_token)
            ]
            cached_name_chunking = f'{hash_sha256(url)}_without_chunking'
        else:
            content = self.split_doc_to_chunk(doc, url, title=title, parser_page_size=parser_page_size)

        time2 = time.time()
        print(f'[file_analyzer] Finished chunking {url} ({title}). Time spent: {time2 - time1} seconds.')

        new_record = Record(url=url, raw=content, title=title).to_dict()
        new_record_str = json.dumps(new_record, ensure_ascii=False)
        self.db.put(cached_name_chunking, new_record_str)
        return new_record

    def split_doc_to_chunk(self,
                           doc: List[dict],
                           path: str,
                           title: str = '',
                           parser_page_size: int = DEFAULT_PARSER_PAGE_SIZE) -> List[Chunk]:
        res = []
        chunk = []
        available_token = parser_page_size
        has_para = False
        for page in doc:
            page_num = page['page_num']
            if not chunk or f'[page: {str(page_num)}]' != chunk[0]:
                chunk.append(f'[page: {str(page_num)}]')
            idx = 0
            len_para = len(page['content'])
            while idx < len_para:
                if not chunk:
                    chunk.append(f'[page: {str(page_num)}]')
                para = page['content'][idx]
                txt = para.get('text', para.get('table'))
                token = para['token']
                if token <= available_token:
                    available_token -= token
                    chunk.append([txt, page_num])
                    has_para = True
                    idx += 1
                else:
                    if has_para:
                        if isinstance(chunk[-1], str) and re.fullmatch(r'^\[page: \d+\]$', chunk[-1]) is not None:
                            chunk.pop()
                        res.append(
                            Chunk(content=PARAGRAPH_SPLIT_SYMBOL.join(
                                [x if isinstance(x, str) else x[0] for x in chunk]),
                                  metadata={
                                      'source': path,
                                      'title': title,
                                      'chunk_id': len(res)
                                  },
                                  token=parser_page_size - available_token))

                        overlap_txt = self._get_last_part(chunk)
                        if overlap_txt.strip():
                            chunk = [f'[page: {str(chunk[-1][1])}]', overlap_txt]
                            has_para = False
                            available_token = parser_page_size - count_tokens(overlap_txt)
                        else:
                            chunk = []
                            has_para = False
                            available_token = parser_page_size
                    else:
                        _sentences = re.split(r'\. |。', txt)
                        sentences = []
                        for s in _sentences:
                            token = count_tokens(s)
                            if not s.strip() or token == 0:
                                continue
                            if token <= available_token:
                                sentences.append([s, token])
                            else:
                                token_list = tokenizer.tokenize(s)
                                for si in range(0, len(token_list), available_token):
                                    ss = tokenizer.convert_tokens_to_string(
                                        token_list[si:min(len(token_list), si + available_token)])
                                    sentences.append([ss, min(available_token, len(token_list) - si)])
                        sent_index = 0
                        while sent_index < len(sentences):
                            s = sentences[sent_index][0]
                            token = sentences[sent_index][1]
                            if not chunk:
                                chunk.append(f'[page: {str(page_num)}]')

                            if token <= available_token or (not has_para):
                                available_token -= token
                                chunk.append([s, page_num])
                                has_para = True
                                sent_index += 1
                            else:
                                assert has_para
                                if isinstance(chunk[-1], str) and re.fullmatch(r'^\[page: \d+\]$',
                                                                               chunk[-1]) is not None:
                                    chunk.pop()
                                res.append(
                                    Chunk(content=PARAGRAPH_SPLIT_SYMBOL.join(
                                        [x if isinstance(x, str) else x[0] for x in chunk]),
                                          metadata={
                                              'source': path,
                                              'title': title,
                                              'chunk_id': len(res)
                                          },
                                          token=parser_page_size - available_token))

                                overlap_txt = self._get_last_part(chunk)
                                if overlap_txt.strip():
                                    chunk = [f'[page: {str(chunk[-1][1])}]', overlap_txt]
                                    has_para = False
                                    available_token = parser_page_size - count_tokens(overlap_txt)
                                else:
                                    chunk = []
                                    has_para = False
                                    available_token = parser_page_size
                        idx += 1
        if has_para:
            if isinstance(chunk[-1], str) and re.fullmatch(r'^\[page: \d+\]$', chunk[-1]) is not None:
                chunk.pop()
            res.append(
                Chunk(content=PARAGRAPH_SPLIT_SYMBOL.join([x if isinstance(x, str) else x[0] for x in chunk]),
                      metadata={
                          'source': path,
                          'title': title,
                          'chunk_id': len(res)
                      },
                      token=parser_page_size - available_token))

        return res

    def _get_last_part(self, chunk: list) -> str:
        overlap = ''
        need_page = chunk[-1][1]
        available_len = 150
        for i in range(len(chunk) - 1, -1, -1):
            if not (isinstance(chunk[i], list) and len(chunk[i]) == 2):
                continue
            if chunk[i][1] != need_page:
                return overlap
            para = chunk[i][0]
            if len(para) <= available_len:
                if overlap:
                    overlap = f'{para}{PARAGRAPH_SPLIT_SYMBOL}{overlap}'
                else:
                    overlap = f'{para}'
                available_len -= len(para)
                continue
            sentence_split_symbol = '. '
            if '。' in para:
                sentence_split_symbol = '。'
            sentences = re.split(r'\. |。', para)
            sentences = [sentence.strip() for sentence in sentences if sentence]
            for j in range(len(sentences) - 1, -1, -1):
                sent = sentences[j]
                if not sent.strip():
                    continue
                if len(sent) <= available_len:
                    if overlap:
                        overlap = f'{sent}{sentence_split_symbol}{overlap}'
                    else:
                        overlap = f'{sent}'
                    available_len -= len(sent)
                else:
                    return overlap
        return overlap


@register_tool('file_analyzer', allow_overwrite=True)
class FileAnalyzer(BaseTool):
    name = 'file_analyzer'
    description = '''Intelligently analyze and extract information from various file types including documents and images. This tool is HIGHLY RECOMMENDED when you need to:
- Extract specific data from PDF, Word (.docx), Excel (.xlsx), PowerPoint files
- Analyze images (JPG, PNG, GIF, etc.) to answer questions about visual content
- Search for particular information within large documents
- Get structured summaries of file contents based on specific goals
- Process files that are too large or complex to read manually

The tool uses advanced AI models to understand both text and visual content, automatically parsing documents and extracting only the relevant information you need. Always use this tool when dealing with files instead of trying to read them directly.'''
    arguments = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file you want to analyze. Supports: PDF (.pdf), Word (.docx, .doc), Excel (.xlsx, .xls), PowerPoint (.pptx, .ppt), text files (.txt, .md, .csv, .json, .xml, .html), and images (.jpg, .png, .gif, .bmp, .webp, .svg). Example: '/path/to/document.pdf' or 'data/report.xlsx'"
            },
            "goal": {
                "type": "string",
                "description": "A clear, specific question or goal describing what information you want to extract from the file. Be as detailed as possible. Examples: 'Find the total revenue in Q3 2024', 'What is the main conclusion of this research paper?', 'Extract all email addresses mentioned', 'What objects are visible in this image?', 'Summarize the key findings in the executive summary section'"
            }
        },
        "required": ["file_path", "goal"]
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.doc_parser = InternalDocParser(cfg)

        self.api_key = os.getenv("FILE_ANALYZER_API_KEY", os.getenv("MAIN_LLM_API_KEY", ""))
        self.api_base = os.getenv("FILE_ANALYZER_API_BASE_URL", os.getenv("MAIN_LLM_API_BASE_URL", ""))
        self.text_model = os.getenv("FILE_ANALYZER_TEXT_MODEL", os.getenv("MAIN_LLM_MODEL", ""))
        self.vision_model = os.getenv("FILE_ANALYZER_VISION_MODEL", "")

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            file_path = params["file_path"]
            goal = params["goal"]
        except Exception as e:
            return f"[file_analyzer] Invalid request format: {str(e)}"

        if not os.path.exists(file_path):
            return f"[file_analyzer] Error: File not found at {file_path}"

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in IMAGE_EXTENSIONS:
            return self.analyze_image(file_path, goal)
        elif file_ext in TEXT_EXTENSIONS:
            return self.analyze_text_file(file_path, goal)
        else:
            return f"[file_analyzer] Error: Unsupported file type '{file_ext}'. Supported types: {TEXT_EXTENSIONS | IMAGE_EXTENSIONS}"

    def analyze_text_file(self, file_path: str, goal: str) -> str:
        try:
            print(f"[file_analyzer] Parsing text file: {file_path}")
            doc_result = self.doc_parser.parse(file_path)

            if 'raw' in doc_result:
                content_parts = []
                for chunk in doc_result['raw']:
                    content_parts.append(chunk['content'])
                file_content = "\n\n".join(content_parts)
            else:
                return "[file_analyzer] Error: Failed to parse document"

            max_content_length = 100000
            if len(file_content) > max_content_length:
                print(f"[file_analyzer] Content too long ({len(file_content)} chars), truncating to {max_content_length}")
                file_content = file_content[:max_content_length]

            print(f"[file_analyzer] Extracting information with LLM...")
            extraction_result = self._extract_with_text_llm(file_content, goal, file_path)

            return extraction_result

        except Exception as e:
            return f"[file_analyzer] Error analyzing text file: {str(e)}"

    def analyze_image(self, file_path: str, goal: str) -> str:
        try:
            print(f"[file_analyzer] Analyzing image: {file_path}")

            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            file_ext = os.path.splitext(file_path)[1].lower()
            mime_type = f"image/{file_ext[1:]}" if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'] else "image/jpeg"

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=60,
            )

            prompt = f"""Analyze this image based on the following goal:

Goal: {goal}

Please provide:
1. Evidence: Describe what you see in the image that is relevant to the goal
2. Summary: Provide a concise answer to the goal based on the image content

Format your response as JSON with "evidence" and "summary" fields."""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chat_completion = client.chat.completions.create(
                        model=self.vision_model,
                        max_tokens=4096,
                        temperature=0.0,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                    )

                    raw_response = chat_completion.choices[0].message.content
                    print(f"[file_analyzer] Vision model response length: {len(raw_response)}")

                    try:
                        result = json.loads(raw_response)
                    except:
                        left = raw_response.find('{')
                        right = raw_response.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            result = json.loads(raw_response[left:right+1])
                        else:
                            result = {
                                "evidence": raw_response,
                                "summary": raw_response
                            }

                    useful_information = f"The useful information in {file_path} for user goal '{goal}' as follows:\n\n"
                    useful_information += f"Evidence in image:\n{result.get('evidence', 'N/A')}\n\n"
                    useful_information += f"Summary:\n{result.get('summary', 'N/A')}\n\n"

                    return useful_information

                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"[file_analyzer] Error calling vision model after {max_retries} attempts: {str(e)}"
                    time.sleep(min(5, 0.5 * (2 ** attempt)))
                    continue

        except Exception as e:
            return f"[file_analyzer] Error analyzing image: {str(e)}"

    def _extract_with_text_llm(self, file_content: str, goal: str, file_path: str, max_retries: int = 3) -> str:

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=60,
        )

        extraction_prompt = TEXT_EXTRACTOR_PROMPT.format(
            file_content=file_content,
            goal=goal
        )

        messages = [{"role": "user", "content": extraction_prompt}]

        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.text_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                )

                raw_response = chat_response.choices[0].message.content

                if len(raw_response) < 10:
                    print(f"[file_analyzer] Response too short, truncating content (attempt {attempt + 1}/{max_retries})")
                    truncate_length = int(0.7 * len(file_content)) if attempt < max_retries - 1 else 25000
                    file_content = file_content[:truncate_length]
                    extraction_prompt = TEXT_EXTRACTOR_PROMPT.format(
                        file_content=file_content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    continue

                try:
                    result = json.loads(raw_response)
                except:
                    left = raw_response.find('{')
                    right = raw_response.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        result = json.loads(raw_response[left:right+1])
                    else:
                        if attempt < max_retries - 1:
                            continue
                        result = {
                            "evidence": "Failed to parse response",
                            "summary": "The file content could not be processed"
                        }

                useful_information = f"The useful information in {file_path} for user goal '{goal}' as follows:\n\n"
                useful_information += f"Evidence in file:\n{result.get('evidence', 'N/A')}\n\n"
                useful_information += f"Summary:\n{result.get('summary', 'N/A')}\n\n"

                return useful_information

            except Exception as e:
                if attempt == max_retries - 1:
                    return f"[file_analyzer] Error calling text LLM after {max_retries} attempts: {str(e)}"
                time.sleep(min(5, 0.5 * (2 ** attempt)))
                continue

        return "[file_analyzer] Failed to extract information after maximum retries"
