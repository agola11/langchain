"""Experiment with different models."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from langchain.chains.base import MultiVariableChain
from langchain.chains.llm import LLMChain
from langchain.input import get_color_mapping, print_text
from langchain.llms.base import LLM
from langchain.prompts.prompt import PromptTemplate


class ModelLaboratory:
    """Experiment with different models."""

    def __init__(
        self, chains: Sequence[MultiVariableChain], names: Optional[List[str]] = None
    ):
        """Initialize with chains to experiment with.

        Args:
            chains: list of chains to experiment with.
        """
        for chain in chains:
            if not isinstance(chain, MultiVariableChain):
                raise ValueError(
                    "ModelLaboratory should now be initialized with SingleVariableChains. "
                    "If you want to initialize with LLMs, use the `from_llms` method "
                    "instead (`ModelLaboratory.from_llms(...)`)"
                )
            if len(chain.input_keys) != 1:
                raise ValueError(
                    "Currently only support chains with one input variable, "
                    f"got {chain.input_keys}"
                )
            if len(chain.output_keys) != 1:
                raise ValueError(
                    "Currently only support chains with one output variable, "
                    f"got {chain.output_keys}"
                )
        if names is not None:
            if len(names) != len(chains):
                raise ValueError("Length of chains does not match length of names.")
        self.chains = chains
        chain_range = [str(i) for i in range(len(self.chains))]
        self.chain_colors = get_color_mapping(chain_range)
        self.names = names

    @classmethod
    def from_llms(
        cls, llms: List[LLM], prompt: Optional[PromptTemplate] = None
    ) -> ModelLaboratory:
        """Initialize with LLMs to experiment with and optional prompt.

        Args:
            llms: list of LLMs to experiment with
            prompt: Optional prompt to use to prompt the LLMs. Defaults to None.
                If a prompt was provided, it should only have one input variable.
        """
        if prompt is None:
            prompt = PromptTemplate(input_variables=["_input"], template="{_input}")
        chains = [LLMChain(llm=llm, prompt=prompt) for llm in llms]
        names = [str(llm) for llm in llms]
        return cls(chains, names=names)

    def compare(self, chain_input: Dict[str, str]) -> None:
        """Compare model outputs on an input text.

        If a prompt was provided with starting the laboratory, then this text will be
        fed into the prompt. If no prompt was provided, then the input text is the
        entire prompt.

        Args:
            chain_input: input dictionary to run all models on.
        """
        print(f"\033[1mInput:\033[0m\n{chain_input}\n")
        for i, chain in enumerate(self.chains):
            if self.names is not None:
                name = self.names[i]
            else:
                name = str(chain)
            print_text(name, end="\n")
            output = chain.run(return_only_outputs=True, **chain_input)
            print_text(str(output), color=self.chain_colors[str(i)], end="\n\n")
