from agents.coordinator_agent import CoordinatorAgent
from agents.existing_reaction_agent import ExistingReactionAgent
from agents.exploring_reaction_agent import ExploringReactionAgent
from agents.summary_agent import SummaryAgent
from agents.critic_agent import CriticAgent
from agents.transferability_agent import TransferabilityAgent


def build_agent_registry(config):
    return {
        "coordinator_agent": CoordinatorAgent(config),
        "existing_reaction_agent": ExistingReactionAgent(config),
        "exploring_reaction_agent": ExploringReactionAgent(config),
        "summary_agent": SummaryAgent(config),
        "critic_agent": CriticAgent(config),
        "transferability_agent": TransferabilityAgent(config),
    }


def get_active_agents(config):
    registry = build_agent_registry(config)
    return {name: registry[name] for name in config.active_agents}