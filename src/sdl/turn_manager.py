import abc
import itertools


class TurnManager(abc.ABC):
	"""
	A class that handles which of a list of users gets to speak in the next dialogue turn.
	"""

	def __init__(self, usernames: list[str], config: dict[str, float]={}):
		"""
		Construct a new TurnManager.

		:param usernames: a list of all usernames in the conversation
		:type usernames: list[str]
		:param config: a dictionary of other configurations, defaults to {}
		:type config: dict[str, float], optional
		"""
		self.usernames = usernames
		self.config = config

	@abc.abstractmethod
	def next_turn_username(self) -> str:
		"""
		Get the username of the next speaker.

		:raises NotImplemented: abstract method
		:return: the next speaker's username
		:rtype: str
		"""
		raise NotImplemented()


class RoundRobbin(TurnManager):
	"""
	A simple turn manager which gives priority to the next user in the queue.
	"""

	def __init__(self, usernames: list[str]):
		super().__init__(usernames)
		self.username_loop = itertools.cycle(self.usernames)
		self.curr_turn = 0

	def next_turn_username(self) -> str:
		return next(self.username_loop)


def turn_manager_factory(turn_manager_type: str, usernames: list[str]) -> TurnManager:
	"""
	A factory which returns a instansiated TurnManager of the type specified by a string.

	:param turn_manager_type: the string specifying the concrete TurnManager class.
	Can be of one of "round_robbin",...
	:type turn_manager_type: TurnManager
	:param usernames: a list of all usernames of each participant in the conversation
	:type usernames: list[str]
	:raises ValueError: if turn_manager_type does not match any classes
	:return: the instansiated TurnManager of the specified type
	:rtype: TurnManager
	"""
	match turn_manager_type.lower():
		case "round_robbin":
			return RoundRobbin(usernames=usernames)
		case _:
			raise ValueError(f"There is no turn manager option called {turn_manager_type}" +
			"Valid values: round_robbin")