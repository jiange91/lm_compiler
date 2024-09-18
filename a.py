from compiler.IR.program import Workflow, hint_possible_destinations, Context
import ast
import inspect

class RewriteDestHintDecorator(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        super().__init__()
        self.old_name = old_name
        self.new_name = new_name
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'hint_possible_destinations':
                # change constant in args to new name
                for i, arg in enumerate(decorator.args[0].elts):
                    if isinstance(arg, ast.Constant) and arg.value == self.old_name:
                        arg.value = self.new_name
        return node

@hint_possible_destinations(['direct_hyde', 'answer_compose', 'answer'])
def final_router(ctx: Context, decision: str):
    if ctx.invoke_time >= 3:
        return 'answer'
    if decision == 'ae' or decision == 're':
        return 'direct_hyde'
    if decision == 'ge':
        return 'answer_compose'
    return 'answer'

source = inspect.getsource(final_router)
tree = ast.parse(source, mode='exec')

transformer = RewriteDestHintDecorator('direct_hyde', 'direct_hyde2')
transformed_tree = transformer.visit(tree)
code = ast.unparse(transformed_tree).strip()
print(code)

# print(ast.dump(tree, indent=4))

# Module(
#     body=[
#         FunctionDef(
#             name='final_router',
#             args=arguments(
#                 posonlyargs=[],
#                 args=[
#                     arg(
#                         arg='ctx',
#                         annotation=Name(id='Context', ctx=Load())),
#                     arg(
#                         arg='decision',
#                         annotation=Name(id='str', ctx=Load()))],
#                 kwonlyargs=[],
#                 kw_defaults=[],
#                 defaults=[]),
#             body=[
#                 If(
#                     test=Compare(
#                         left=Attribute(
#                             value=Name(id='ctx', ctx=Load()),
#                             attr='invoke_time',
#                             ctx=Load()),
#                         ops=[
#                             GtE()],
#                         comparators=[
#                             Constant(value=3)]),
#                     body=[
#                         Return(
#                             value=Constant(value='answer'))],
#                     orelse=[]),
#                 If(
#                     test=BoolOp(
#                         op=Or(),
#                         values=[
#                             Compare(
#                                 left=Name(id='decision', ctx=Load()),
#                                 ops=[
#                                     Eq()],
#                                 comparators=[
#                                     Constant(value='ae')]),
#                             Compare(
#                                 left=Name(id='decision', ctx=Load()),
#                                 ops=[
#                                     Eq()],
#                                 comparators=[
#                                     Constant(value='re')])]),
#                     body=[
#                         Return(
#                             value=Constant(value='direct_hyde'))],
#                     orelse=[]),
#                 If(
#                     test=Compare(
#                         left=Name(id='decision', ctx=Load()),
#                         ops=[
#                             Eq()],
#                         comparators=[
#                             Constant(value='ge')]),
#                     body=[
#                         Return(
#                             value=Constant(value='answer_compose'))],
#                     orelse=[]),
#                 Return(
#                     value=Constant(value='answer'))],
#             decorator_list=[
#                 Call(
#                     func=Name(id='hint_possible_destinations', ctx=Load()),
#                     args=[
#                         List(
#                             elts=[
#                                 Constant(value='direct_hyde'),
#                                 Constant(value='answer_compose'),
#                                 Constant(value='answer')],
#                             ctx=Load())],
#                     keywords=[])],
#             type_params=[])],
#     type_ignores=[])