from ..utils import Logger, Color
from ..config import Config
import dspy


class RagSignature(dspy.Signature):
    documents: list[str] = dspy.InputField(
        description='List of retrieved documents relevant to the question.'
    )
    question: str = dspy.InputField(
        description='The question to be answered based on the retrieved'
        ' documents.'
    )
    answer: str = dspy.OutputField(
        description='Answer a little sentence based on documents. Don\'t write'
        ' Markdown, just the answer.'
    )


class DspyInterface:
    logger: Logger
    app_config: Config

    lm: dspy.LM

    predict: dspy.Predict

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

        self.predict = dspy.Predict(RagSignature)
