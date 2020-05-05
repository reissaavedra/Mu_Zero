from abc import ABC, abstractmethod


class AbstractGame(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def perform_step(self, action):
        """
            Action to the game
        """
        pass

    @staticmethod
    def get_current_player():
        """
        :return: current player
        """
        return 0

    @abstractmethod
    def get_legal_actions(self):
        """
        Define and return legal actions
        :return: legal actions
        """
        pass

    def close(self):
        """
        Close the game
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display the game observation
        """
        pass

    def get_human_action(self):
        """
        Ask the user for a legal action, and return corresponding number
        :return:
        """
        choice = input(
            "Ingrese la acción a realizar por el jugador {}: ".format(self.get_current_player())
        )
        while choice not in [str(action) for action in self.get_legal_actions()]:
            choice = input("Acción no válida. Ingrese otra opción: ")
        return int(choice)

    def get_action_expert_agent(self):
        """
        Action that hard coded agent MuZero faces to assess his progress in games.
        :return:
        """
        raise NotImplementedError

    def convert_action_to_string(self, action_number):
        """
        Convert action to representing string
        :return: Representing string from action
        """
        return str(action_number)




