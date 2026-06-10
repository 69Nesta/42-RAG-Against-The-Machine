from ..models import ChunkContentModel
from ..utils import Logger, Color
from ..config import Config

import dspy


class AnswerSignature(dspy.Signature):
    documents: list[ChunkContentModel] = dspy.InputField(
        description='List of retrieved documents relevant to the question.'
    )
    question: str = dspy.InputField(
        description='The question to be answered based on the retrieved '
        'documents.'
    )
    answer: str = dspy.OutputField(
        description=(
            'Answer the question in 1-3 sentences based only on the provided '
            'documents. At the end, include the sources path used in the '
            'format \'Sources: <path_source1>, <path_source2>, ...\'. '
            'Do not use Markdown.'
        )
    )


class ExpandQuery(dspy.Signature):
    '''
    Generate alternative search queries that approach the topic
    from different angles: synonyms, related concepts, more specific
    or more general formulations. Avoid repeating words from the
    original query.
    '''

    query: str = dspy.InputField()

    bm25_keywords: list[str] = dspy.OutputField(
        description=(
            'Domain-specific technical terms related to the query topic. '
            'Include: synonyms, component names, related concepts, common '
            'abbreviations. Do NOT repeat words already present in the query. '
            'Output 5-8 individual terms or short phrases.'
        )
    )
    semantic_queries: list[str] = dspy.OutputField(
        description=(
            '2-3 queries that approach the same information need from '
            'different angles: one more general, one more specific, one using'
            ' different vocabulary. Avoid reusing the exact phrasing of the'
            ' original query.'
        )
    )


class HyDESignature(dspy.Signature):
    '''
    Write a short passage (2-3 sentences) that directly answers the question.
    '''

    question: str = dspy.InputField()
    hypothetical_passage: str = dspy.OutputField(
        desc=(
            'A concise factual passage, 2-3 sentences, as if from a reference'
            ' document'
        )
    )


class DspyInterface:
    logger: Logger
    app_config: Config

    lm: dspy.LM

    answer_predict: dspy.Predict
    expand_query_predict: dspy.Predict

    def __init__(self, config: Config) -> None:
        self.app_config = config
        self.logger = Logger('DspyInterface', Color.MAGENTA, config.verbose)

        self.lm = dspy.LM(
            model=config.model_name,
            api_base=config.api_base,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cache=config.dspy_cache,
        )
        dspy.configure(lm=self.lm)

        self.answer_predict = dspy.Predict(AnswerSignature)
        self.expand_query_predict = dspy.Predict(ExpandQuery)
        self.hyde_predict = dspy.Predict(HyDESignature)
