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
from typing import Any

from openai import AsyncAzureOpenAI

from .client import EmbedderClient

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedderClient(EmbedderClient):
    """Wrapper class for AsyncAzureOpenAI that implements the EmbedderClient interface."""

    def __init__(self, azure_client: AsyncAzureOpenAI, model: str = 'text-embedding-3-small'):
        self.azure_client = azure_client
        self.model = model

    async def create(self, input_data: str | list[str] | Any) -> list[float]:
        """Create embeddings using Azure OpenAI client."""
        logger.debug(f"Creating Azure OpenAI embedding for input type: {type(input_data)}, model: {self.model}")
        try:
            # Handle different input types
            if isinstance(input_data, str):
                text_input = [input_data]
                logger.debug(f"Converted string input to list, length: {len(text_input)}")
            elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                text_input = input_data
                logger.debug(f"Using list input directly, length: {len(text_input)}")
            else:
                # Convert to string list for other types
                text_input = [str(input_data)]
                logger.debug(f"Converted {type(input_data)} input to string list")

            logger.debug(f"Calling Azure OpenAI embeddings API with model: {self.model}")
            response = await self.azure_client.embeddings.create(model=self.model, input=text_input)
            
            logger.debug(f"Azure OpenAI API response type: {type(response)}")
            logger.debug(f"Azure OpenAI API response has .data attr: {hasattr(response, 'data')}")
            if hasattr(response, 'data'):
                logger.debug(f"Response data type: {type(response.data)}, length: {len(response.data) if response.data else 0}")
                if response.data and len(response.data) > 0:
                    logger.debug(f"First embedding type: {type(response.data[0])}, has embedding attr: {hasattr(response.data[0], 'embedding')}")
            else:
                logger.error(f"Azure OpenAI API returned unexpected response format: {response}")
                logger.error(f"Response content: {str(response)[:500]}")
                raise ValueError(f"Azure OpenAI API returned string instead of embedding object: {str(response)[:200]}")

            # Return the first embedding as a list of floats
            return response.data[0].embedding
        except Exception as e:
            logger.error(f'Error in Azure OpenAI embedding: {e}')
            logger.error(f"Input data type: {type(input_data)}")
            logger.error(f"Model: {self.model}")
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create batch embeddings using Azure OpenAI client."""
        logger.debug(f"Creating Azure OpenAI batch embeddings for {len(input_data_list)} inputs, model: {self.model}")
        try:
            logger.debug(f"Calling Azure OpenAI batch embeddings API with model: {self.model}")
            response = await self.azure_client.embeddings.create(
                model=self.model, input=input_data_list
            )

            logger.debug(f"Azure OpenAI batch API response type: {type(response)}")
            logger.debug(f"Azure OpenAI batch API response has .data attr: {hasattr(response, 'data')}")
            if hasattr(response, 'data'):
                logger.debug(f"Batch response data type: {type(response.data)}, length: {len(response.data) if response.data else 0}")
            else:
                logger.error(f"Azure OpenAI batch API returned unexpected response format: {response}")
                logger.error(f"Response content: {str(response)[:500]}")
                raise ValueError(f"Azure OpenAI batch API returned string instead of embedding object: {str(response)[:200]}")

            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f'Error in Azure OpenAI batch embedding: {e}')
            logger.error(f"Input data list length: {len(input_data_list)}")
            logger.error(f"Model: {self.model}")
            raise
