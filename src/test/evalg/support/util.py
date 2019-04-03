from unittest import TestCase


class NodeCheckTestCase(TestCase):

    def _check_node(self, node, value, left_value, right_value, parent_value):
        self.assertEqual(value, node.value, 'Node values not equal')
        if not node.left:
            self.assertIsNone(left_value, 'Left value not none')
        else:
            self.assertEqual(left_value, node.left.value, 'Left values not equal')

        if not node.right:
            self.assertIsNone(right_value, 'Right value not none')
        else:
            self.assertEqual(right_value, node.right.value, 'Right values not equal')

        if not node.parent:
            self.assertIsNone(parent_value, 'Parent value not none')
        else:
            self.assertEqual(parent_value, node.parent.value, 'Parent values not equal')

    def check_root(self, parent_node, value, left_value, right_value):
        self._check_node(parent_node, value, left_value, right_value, None)

    def check_leaf(self, leaf_node, value, parent_value):
        self._check_node(leaf_node, value, None, None, parent_value)

    def check_stump(self, stump_node, value):
        self._check_node(stump_node, value, None, None, None)
