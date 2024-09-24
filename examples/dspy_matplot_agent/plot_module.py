import logging
import os
import uuid

from agents.query_expansion_agent import QueryExpansionAgent, QueryExpansionModule
from agents.plot_agent.agent import PlotCoder, PlotRefiner, PlotAgentModule, Debugger
from agents.visual_refine_agent import VisualRefineAgent
import dspy

class MatPlotModule(dspy.Module):
    def __init__(self, model_type):
        self.model_type = model_type
        self.query_expansion_agent = QueryExpansionModule(model_type=model_type)
        
        self.plot_coder = dspy.Predict(PlotCoder)
        self.plot_debugger = dspy.Predict(Debugger)
        self.plot_refiner = dspy.Predict(PlotRefiner)
        
        self.initial_plot_agent = PlotAgentModule(
            coder=self.plot_coder,
            debugger=self.plot_debugger,
            model_type=model_type,
        )
        self.refine_plot_agent = PlotAgentModule(
            coder=self.plot_refiner,
            debugger=self.plot_debugger,
            model_type=model_type,
        )
    
    def forward(
        self,
        query,
        directory_path,
        example_id,
        input_path,
    ):
        # Prepare workspace
        workspace = f'{directory_path}/{example_id}_{uuid.uuid4().hex}'
        if not os.path.exists(workspace):
            # If it doesn't exist, create the directory
            os.makedirs(workspace, exist_ok=True)
            if os.path.exists(input_path):
                os.system(f'cp -r {input_path}/* {workspace}')
        else:
            logging.info(f"Directory '{workspace}' already exists.")

        logging.info('=========Query Expansion AGENT=========')
        config = {'workspace': workspace}
        expanded_simple_instruction = self.query_expansion_agent.run(query)
        logging.info('=========Expanded Simple Instruction=========')
        logging.info(expanded_simple_instruction)
        logging.info('=========Plotting=========')

        # Initial plotting
        logging.info('=========Novice 4 Plotting=========')
        novice_log, novice_code = self.initial_plot_agent.run(
            query=query,
            expanded_query=expanded_simple_instruction,
            query_type='initial',
            file_name='novice.png',
            workspace=workspace,
        )
        logging.info(novice_log)
        # logging.info('=========Original Code=========')
        # logging.info(novice_code)

        # Visual refinement
        if os.path.exists(f'{workspace}/novice.png'):
            print('Use original code for visual feedback')
            visual_refine_agent = VisualRefineAgent('novice.png', config, '', query)
            visual_feedback = visual_refine_agent.run(self.model_type, 'novice', 'novice_final.png')
            logging.info('=========Visual Feedback=========')
            logging.info(visual_feedback)
            final_instruction = '' + '\n\n' + visual_feedback
            
            novice_log, novice_code = self.refine_plot_agent.run(
                query=query,
                expanded_query=final_instruction,
                query_type='refinement',
                file_name='novice_final.png',
                workspace=workspace,
            )
            logging.info(novice_log)
        
        logging.info('=========query expansion Usage=========')
        usage = self.query_expansion_agent.engine.get_usage_and_reset()
        logging.info(usage)
        logging.info('=========initial plotter Usage=========')
        usage = self.initial_plot_agent.engine.get_usage_and_reset()
        logging.info(usage)
        logging.info('=========refine plotter Usage=========')
        usage = self.refine_plot_agent.engine.get_usage_and_reset()
        logging.info(usage)
        
        return dspy.Prediction(
            img_path=f'{workspace}/novice_final.png',
            rollback=f'{workspace}/novice.png',
        )