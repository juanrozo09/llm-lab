class ProviderUnavailableError(Exception):
    def __init__(self, provider: str, original_error: Exception) -> None:
        self.provider = provider
        self.original_error = original_error
        super().__init__(
            f"{provider} provider unavailable after retries: {original_error}"
        )
