from abc import abstractmethod
from torch_spyre._inductor.operation import Operation

class SpyreLxOptimizationPass:
    @abstractmethod
    def apply_pass(self, operations: list[Operation]) -> list[Operation]:
        """
        _summary_

        Args:
            operations (list[Operation]): _description_

        Returns:
            list[Operation]: _description_
        """        
        pass