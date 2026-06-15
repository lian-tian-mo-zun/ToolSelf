from __future__ import annotations
import copy
import re
from typing import Protocol, Literal
from abc import abstractmethod


HistoryItem = dict[str, any]
History = list[HistoryItem]


class AbstractHistoryProcessor(Protocol):

    @abstractmethod
    def __call__(self, history: History) -> History:
        raise NotImplementedError


class DefaultHistoryProcessor:

    def __init__(self):
        self.type = "default"

    def __call__(self, history: History) -> History:
        return history


class LastNObservations:

    def __init__(
        self,
        n: int = 5,
        elide_message_template: str = "Old environment output: ({n_lines} lines omitted)",
        always_keep_system: bool = True,
        always_keep_first_user: bool = True,
    ):
        if n <= 0:
            raise ValueError("n must be a positive integer")

        self.type = "last_n_observations"
        self.n = n
        self.elide_message_template = elide_message_template
        self.always_keep_system = always_keep_system
        self.always_keep_first_user = always_keep_first_user

    def _count_lines(self, content: str) -> int:
        if not isinstance(content, str):
            return 0
        return len(content.splitlines())

    def _is_observation(self, entry: HistoryItem) -> bool:
        if entry.get("role") != "user":
            return False

        content = entry.get("content", "")
        if "User:" in content or "Question:" in content:
            return False

        if "<tool_response>" in content or "Observation:" in content:
            return True

        return False

    def _get_observation_indices(self, history: History) -> list[int]:
        observation_indices = []
        first_user_seen = False

        for idx, entry in enumerate(history):
            if entry.get("role") == "system":
                continue

            if entry.get("role") == "user" and not first_user_seen:
                first_user_seen = True
                continue

            if self._is_observation(entry):
                observation_indices.append(idx)

        return observation_indices

    def __call__(self, history: History) -> History:
        if not history:
            return history

        new_history = []
        observation_indices = self._get_observation_indices(history)

        if len(observation_indices) > self.n:
            omit_indices = set(observation_indices[:-self.n])
        else:
            omit_indices = set()

        first_user_idx = None
        for idx, entry in enumerate(history):
            if entry.get("role") == "user":
                first_user_idx = idx
                break

        for idx, entry in enumerate(history):
            if self.always_keep_system and entry.get("role") == "system":
                new_history.append(entry)
                continue

            if self.always_keep_first_user and idx == first_user_idx:
                new_history.append(entry)
                continue

            if idx in omit_indices:
                data = copy.deepcopy(entry)
                n_lines = self._count_lines(entry.get("content", ""))
                data["content"] = self.elide_message_template.format(n_lines=n_lines)
                new_history.append(data)
            else:
                new_history.append(entry)

        return new_history


class RemoveThinkTags:

    def __init__(self, keep_last_n: int = 0):
        self.type = "remove_think_tags"
        self.keep_last_n = keep_last_n
        self._pattern = re.compile(r'<think>.*?</think>', re.DOTALL)

    def __call__(self, history: History) -> History:
        new_history = []

        for idx, entry in enumerate(reversed(history)):
            data = copy.deepcopy(entry)

            if data.get("role") == "assistant":
                if idx < self.keep_last_n:
                    new_history.append(data)
                else:
                    content = data.get("content", "")
                    if isinstance(content, str):
                        data["content"] = self._pattern.sub('', content).strip()
                    new_history.append(data)
            else:
                new_history.append(data)

        return list(reversed(new_history))


class TruncateObservations:

    def __init__(
        self,
        max_length: int = 10000,
        truncate_template: str = "\n... (truncated {n_chars} characters) ...\n",
    ):
        self.type = "truncate_observations"
        self.max_length = max_length
        self.truncate_template = truncate_template

    def __call__(self, history: History) -> History:
        new_history = []

        for entry in history:
            data = copy.deepcopy(entry)
            content = data.get("content", "")

            if isinstance(content, str) and len(content) > self.max_length:
                n_chars = len(content) - self.max_length
                truncated = (
                    content[:self.max_length] +
                    self.truncate_template.format(n_chars=n_chars)
                )
                data["content"] = truncated

            new_history.append(data)

        return new_history


class TruncationHistoryProcessor:
    def __init__(
        self,
        last_n_observations: int = 5,
        max_observation_length: int = 10000,
        remove_think_tags: bool = False,
    ):
        self.type = "truncation"
        self.processors = []

        if max_observation_length > 0:
            self.processors.append(TruncateObservations(max_length=max_observation_length))

        if last_n_observations > 0:
            self.processors.append(LastNObservations(n=last_n_observations))

        if remove_think_tags:
            self.processors.append(RemoveThinkTags())

    def __call__(self, history: History) -> History:
        processed = history
        for processor in self.processors:
            processed = processor(processed)
        return processed


def create_history_processor(
    processor_type: Literal["default", "truncation", "last_n"] = "default",
    **kwargs
) -> AbstractHistoryProcessor:
    if processor_type == "default":
        return DefaultHistoryProcessor()
    elif processor_type == "truncation":
        return TruncationHistoryProcessor(**kwargs)
    elif processor_type == "last_n":
        return LastNObservations(**kwargs)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
