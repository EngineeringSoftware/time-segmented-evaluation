import collections
import copy
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple

from seutil import BashUtils, IOUtils, LoggingUtils
from tqdm import tqdm

from tseval.Environment import Environment

logger = LoggingUtils.get_logger(__name__)


class ModelUtils:

    TOKENIZE_JAVAPARSER_BATCH_SIZE = 10000

    @classmethod
    def get_random_seed(cls) -> int:
        """
        Generates a random int as seed, within the range of [0, 2**32)
        The seed is generated based on current time
        """
        return time.time_ns() % (2**32)

    @classmethod
    def tokenize_javaparser(cls, code: str) -> List[str]:
        return cls.tokenize_javaparser_batch([code])[0]

    @classmethod
    def tokenize_javaparser_batch(
            cls,
            code_list: List[str],
            dup_share: bool = True,
            tbar: Optional[tqdm] = None,
    ) -> List[List[str]]:
        """
        Tokenizes a list of code using JavaParser.
        :param code_list: a list of code to be tokenized.
        :param dup_share: if True (default), the returned lists of tokens will be shared across duplicate code
            (thus modifying one of them will affect others).
        :param tbar: optional tqdm progress bar.
        :returns a list of tokenized code.
        """
        # get an unique list of code to tokenize, maintain a backward mapping
        code_2_id: OrderedDict[str, int] = collections.OrderedDict()
        ids: List[int] = []
        for c in code_list:
            ids.append(code_2_id.setdefault(c, len(code_2_id)))

        unique_code_list = list(code_2_id.keys())
        if tbar is not None:
            tbar.set_description(f"JavaParser Tokenize ({len(unique_code_list)}U/{len(code_list)})")
            tbar.reset(len(unique_code_list))

        # Tokenize (with batching)
        unique_tokens_list = []
        for beg in range(0, len(unique_code_list), cls.TOKENIZE_JAVAPARSER_BATCH_SIZE):
            unique_tokens_list += cls.tokenize_javaparser_batch_(
                unique_code_list[beg:beg + cls.TOKENIZE_JAVAPARSER_BATCH_SIZE], tbar=tbar,
            )

        if dup_share:
            return [unique_tokens_list[i] for i in ids]
        else:
            return [copy.copy(unique_tokens_list[i]) for i in ids]

    @classmethod
    def tokenize_javaparser_batch_(
            cls,
            code_list: List[str],
            tbar: Optional[tqdm] = None,
    ):
        # Use JavaParser to tokenize
        Environment.require_collector()

        tokenizer_inputs = []
        for code in code_list:
            tokenizer_inputs.append({
                "index": len(tokenizer_inputs),
                "code": code,
            })

        inputs_file = Path(tempfile.mktemp())
        IOUtils.dump(inputs_file, tokenizer_inputs, IOUtils.Format.json)
        outputs_file = Path(tempfile.mktemp())

        BashUtils.run(
            f"java -cp {Environment.collector_jar} org.tseval.ExtractToken '{inputs_file}' '{outputs_file}'",
            expected_return_code=0,
        )

        tokenizer_outputs = IOUtils.load(outputs_file, IOUtils.Format.json)
        IOUtils.rm(inputs_file)
        IOUtils.rm(outputs_file)

        # Check for tokenizer failures
        for code, output in zip(code_list, tokenizer_outputs):
            if len(code.strip()) == 0:
                logger.warning(f"Empty code: {code}")
                continue
            if len(output["tokens"]) == 0:
                logger.warning(f"Tokenizer failed: {code}")

        if tbar is not None:
            tbar.update(len(code_list))

        return [d["tokens"] for d in tokenizer_outputs]

    RE_SUBTOKENIZE = re.compile(r"(?<=[_$])(?!$)|(?<!^)(?=[_$])|(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[0-9]|[A-Z][a-z0-9])|(?<=[0-9])(?=[a-zA-Z])")
    SPACE_TOKEN = "<SPACE>"

    @classmethod
    def is_identifier(cls, token: str) -> bool:
        return len(token) > 0 and \
                (token[0].isalpha() or token[0] in "_$") and \
                all([c.isalnum() or c in "_$" for c in token])

    @classmethod
    def subtokenize(cls, token: str) -> List[str]:
        """
        Subtokenizes an identifier name into subtokens, by CamelCase and snake_case.
        """
        # Only subtokenize identifier words (starts with letter _$, contains only alnum and _$)
        if cls.is_identifier(token):
            return cls.RE_SUBTOKENIZE.split(token)
        else:
            return [token]

    @classmethod
    def subtokenize_batch(cls, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """
        Subtokenizes list of tokens.
        :return a list of subtokens, and a list of pointers to the original token indices.
        """
        sub_tokens = []
        src_indices = []
        for i, token in enumerate(tokens):
            new_sub_tokens = cls.subtokenize(token)
            sub_tokens += new_sub_tokens
            src_indices += [i] * len(new_sub_tokens)
        return sub_tokens, src_indices

    @classmethod
    def subtokenize_space_batch(cls, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """
        Subtokenizes list of tokens, and inserts special <SPACE> token when necessary
        (between two identifiers).
        :return a list of subtokens, and a list of pointers to the original token indices.
        """
        sub_tokens = []
        src_indices = []
        last_is_identifier = False
        for i, token in enumerate(tokens):
            is_identifier = cls.is_identifier(token)
            if last_is_identifier and is_identifier:
                sub_tokens.append(cls.SPACE_TOKEN)
                src_indices.append(-1)
            new_sub_tokens = cls.subtokenize(token)
            sub_tokens += new_sub_tokens
            src_indices += [i] * len(new_sub_tokens)
            last_is_identifier = is_identifier
        return sub_tokens, src_indices

    @classmethod
    def regroup_subtokens(cls, subtokens: List[str], src_indices: List[int]) -> List[str]:
        """
        Given a list of subtokens and the original token indices, groups them back to tokens.
        :param subtokens: a list of subtokens.
        :param src_indices: the i-th indice should point to the original token that
            sub_tokens[i] belongs to; -1 means it is a special sub_token.
        :return: a list of tokens after regrouping.
        """
        id2tokens: Dict[int, str] = collections.defaultdict(str)
        for subtoken, i in zip(subtokens, src_indices):
            if i >= 0:
                id2tokens[i] += subtoken
        return [id2tokens[i] for i in sorted(id2tokens.keys())]
