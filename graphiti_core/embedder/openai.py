"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections.abc import Iterable

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'


class OpenAIEmbedderConfig(EmbedderConfig):
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None


class OpenAIEmbedder(EmbedderClient):
    """
    OpenAI Embedder Client

    This client supports both AsyncOpenAI and AsyncAzureOpenAI clients.
    """

    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        if config is None:
            config = OpenAIEmbedderConfig()
        self.config = config

        if client is not None:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        logger.debug(f"Creating embedding for input type: {type(input_data)}, model: {self.config.embedding_model}")
        result = None
        try:
            result = await self.client.embeddings.create(
                input=input_data, model=self.config.embedding_model
            )
            if isinstance(result, str):
                logger.error(
                    "OpenAI-compatible server returned a string, not an object:"
                    f" {result}"
                )
                raise ValueError(
                    f"Expected embedding object, but got a string: {result[:200]}"
                )
            logger.debug(f"OpenAI API response type: {type(result)}")
            logger.debug(f"OpenAI API response has .data attr: {hasattr(result, 'data')}")
            if hasattr(result, 'data'):
                logger.debug(f"Response data type: {type(result.data)}, length: {len(result.data) if result.data else 0}")
                if result.data and len(result.data) > 0:
                    logger.debug(f"First embedding type: {type(result.data[0])}, has embedding attr: {hasattr(result.data[0], 'embedding')}")
            else:
                logger.error(f"OpenAI API returned unexpected response format: {result}")
                logger.error(f"Response content: {str(result)[:500]}")
                raise ValueError(f"OpenAI API returned string instead of embedding object: {str(result)[:200]}")
            
            return result.data[0].embedding[: self.config.embedding_dim]
        except Exception as e:
            logger.error(f"Error in OpenAI embedding creation: {e}, raw response: {result}")
            logger.error(f"Input data type: {type(input_data)}")
            logger.error(f"Model: {self.config.embedding_model}")
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        logger.debug(f"Creating batch embeddings for {len(input_data_list)} inputs, model: {self.config.embedding_model}")
        result = None
        try:
            result = await self.client.embeddings.create(
                input=input_data_list, model=self.config.embedding_model
            )
            if isinstance(result, str):
                logger.error(
                    "OpenAI-compatible server returned a string, not an object:"
                    f" {result}"
                )
                raise ValueError(
                    f"Expected embedding object, but got a string: {result[:200]}"
                )
            logger.debug(f"OpenAI batch API response type: {type(result)}")
            logger.debug(f"OpenAI batch API response has .data attr: {hasattr(result, 'data')}")
            if hasattr(result, 'data'):
                logger.debug(f"Batch response data type: {type(result.data)}, length: {len(result.data) if result.data else 0}")
            else:
                logger.error(f"OpenAI batch API returned unexpected response format: {result}")
                logger.error(f"Response content: {str(result)[:500]}")
                raise ValueError(f"OpenAI batch API returned string instead of embedding object: {str(result)[:200]}")
            
            return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]
        except Exception as e:
            logger.error(f"Error in OpenAI batch embedding creation: {e}, raw response: {result}")
            logger.error(f"Input data list length: {len(input_data_list)}")
            logger.error(f"Model: {self.config.embedding_model}")
            raise
