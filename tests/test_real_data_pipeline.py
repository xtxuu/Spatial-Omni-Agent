"""Check real-data entry-point handoff without network or production exports."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import run_spatial_agent as pipeline


class RealPipelineTests(unittest.TestCase):
    def invoke(self, data_result, analysis_result=None, execute=False, replies=None):
        with TemporaryDirectory() as directory:
            store = MagicMock()
            data = MagicMock()
            if isinstance(data_result, list):
                data.run.side_effect = data_result
            else:
                data.run.return_value = data_result
            analysis = MagicMock()
            if isinstance(analysis_result, list):
                analysis.run.side_effect = analysis_result
            else:
                analysis.run.return_value = analysis_result
            answers = iter(replies) if replies is not None else None
            def respond(prompt):
                answer = next(answers)
                if isinstance(answer, BaseException):
                    raise answer
                return answer
            with patch.object(pipeline, 'SpatialStore', return_value=store), \
                 patch.object(pipeline, 'GeminiDataAgent', return_value=data), \
                 patch.object(pipeline, 'SpatialAnalysisAgent', return_value=analysis) as factory, \
                 patch.object(pipeline, 'load_plan_gene_lists', return_value=[]) as reader:
                result = pipeline.run_pipeline('Compare infarction against control with DEG only',
                    dataset=Path(directory)/'real_release', output_root=Path(directory)/'runs',
                    key_file='not-read-in-test', model='test-model', execute=execute,
                    reply=respond if answers is not None else None)
                self.assertTrue((Path(result['run_root'])/'pipeline_result.json').is_file())
                data.close.assert_called_once()
                return result, data, analysis, factory, reader

    def test_single_question_reaches_both_agents_without_automatic_execution(self):
        result, data, analysis, factory, reader = self.invoke(
            {'status':'ready','artifacts':[{'manifest_uri':'real_export/manifest.json'}]},
            {'status':'planned','jobs':[]})
        self.assertEqual(result['status'],'planned')
        question=analysis.run.call_args.args[0]
        self.assertIn(question, data.run.call_args.args[0])
        self.assertFalse(analysis.run.call_args.kwargs['execute'])
        self.assertEqual(factory.call_args.args[0],'real_export/manifest.json')
        self.assertEqual(factory.call_args.kwargs['defaults'],{'gene_alignment':'canonical'})
        reader.assert_not_called()
        analysis.close.assert_called_once()

    def test_clarification_or_multiple_exports_never_start_analysis(self):
        for data_result in [
            {'status':'clarification_required','message':'Which study?','artifacts':[]},
            {'status':'ready','artifacts':[{'manifest_uri':'a'},{'manifest_uri':'b'}]},
            {'status':'error','artifacts':[{'manifest_uri':'partial'}]},
        ]:
            result, _, _, factory, _ = self.invoke(data_result)
            factory.assert_not_called()
            self.assertIsNone(result['analysis'])

    def test_explicit_execute_hands_off_both_directions(self):
        result, _, analysis, _, reader = self.invoke(
            {'status':'ready','artifacts':[{'manifest_uri':'real_export/manifest.json'}]},
            {'status':'partial_success','plan_manifest_uri':'plan/manifest.json',
             'analysis_manifest_uris':['child/manifest.json']}, execute=True)
        self.assertTrue(analysis.run.call_args.kwargs['execute'])
        self.assertEqual([c.args[1] for c in reader.call_args_list],
                         ['significant_positive','significant_negative'])
        self.assertIn('downstream_gene_lists_uri',result)

    def test_data_clarification_keeps_session_and_passes_choice_to_analysis(self):
        result, data, analysis, factory, _ = self.invoke([
            {'status':'clarification_required','message':'Which study?','artifacts':[]},
            {'status':'ready','artifacts':[{'manifest_uri':'brain/manifest.json'}]},
        ], {'status':'planned'}, replies=['', 'GSE279181, MS versus Control'])
        self.assertEqual(result['status'],'planned')
        self.assertEqual(data.run.call_count,2)
        self.assertIn('GSE279181',data.run.call_args.args[0])
        self.assertIn('GSE279181',analysis.run.call_args.args[0])
        factory.assert_called_once()

    def test_analysis_clarification_reuses_bundle_and_agent(self):
        result, data, analysis, factory, _ = self.invoke(
            {'status':'ready','artifacts':[{'manifest_uri':'brain/manifest.json'}]}, [
                {'status':'clarification_required','message':'Which cell type?'},
                {'status':'planned'},
            ], replies=['Astrocyte'])
        self.assertEqual(result['status'],'planned')
        data.run.assert_called_once()
        self.assertEqual(analysis.run.call_count,2)
        self.assertEqual(analysis.run.call_args.args[0],'Astrocyte')
        factory.assert_called_once()

    def test_user_can_quit_without_analysis_or_new_api_call(self):
        result, data, _, factory, _ = self.invoke(
            {'status':'clarification_required','message':'Which study?','artifacts':[]},
            replies=['/quit'])
        self.assertEqual(result['conversation_end'],'user_exit')
        self.assertEqual(result['status'],'clarification_required')
        data.run.assert_called_once()
        factory.assert_not_called()

    def test_input_closed_saves_receipt(self):
        result, data, _, factory, _ = self.invoke(
            {'status':'clarification_required','message':'Which study?','artifacts':[]},
            replies=[EOFError()])
        self.assertEqual(result['conversation_end'],'input_closed')
        data.run.assert_called_once()
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
