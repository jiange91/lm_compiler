from queue import Queue

class TreeNode:
    def __init__(self, parent, config, score):
        self.parent = parent
        self.config = config
        self.score = score
        self.next_steps = []
    
    def __repr__(self) -> str:
        return f'{{config: {self.config}, score: {self.score}}}'
        

class ScoreTree:
    """
    This class store the estimation of all possible config path for a program
    """
    def __init__(self, root_config, root_score):
        self.root = TreeNode(None, root_config, root_score)
        self.exixt_nodes = []
        
    def add_new_estimation(
        self, 
        parent: TreeNode, 
        config, 
        score,
        is_exist: bool
    ):
        new_node = TreeNode(parent, config, score)
        parent.next_steps.append(new_node)
        if is_exist:
            self.exixt_nodes.append(new_node)
        return new_node

            
    def get_path(self, predicate: callable):
        """
        Get all paths that have score within the gap
        """
        paths = []
        for node in self.exixt_nodes:
            if predicate(node):
                path = []
                while node is not None:
                    path.append(node)
                    node = node.parent
                paths.append(path[::-1])
        return paths
                