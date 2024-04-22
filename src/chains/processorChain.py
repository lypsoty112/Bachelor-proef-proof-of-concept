from src.chains.baseChain import BaseChain


class ProcessorChain(BaseChain):
    def __init__(self):
        super().__init__()

    def build(self) -> None:
        super().build()

    def run(self, data: object) -> object:
        return super().run(data)

    async def run_async(self, data: object) -> object:
        return await super().run_async(data)

    def pre_run(self, data: object) -> object:
        return super().pre_run(data)

    def post_run(self, data: object) -> object:
        return super().post_run(data)
