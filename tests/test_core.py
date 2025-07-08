import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import PromptEnhancer, get_ollama_models, call_ollama, choose_enhancement_strategy, ENHANCEMENT_PATTERNS

class TestPromptEnhancer(unittest.TestCase):

    def setUp(self):
        self.enhancer = PromptEnhancer()

    def test_analyze_prompt(self):
        prompt = "Please give me an example of a good prompt."
        analysis = self.enhancer.analyze_prompt(prompt)
        self.assertIn('length', analysis)
        self.assertIn('clarity_score', analysis)
        self.assertIn('structure_score', analysis)
        self.assertIn('specificity_score', analysis)
        self.assertIn('suggestions', analysis)

    def test_enhance_prompt(self):
        prompt = "test prompt"
        enhancement_type = "xml_structure"
        enhanced_prompt = self.enhancer.enhance_prompt(prompt, enhancement_type)
        self.assertIn(prompt, enhanced_prompt)
        self.assertIn("<instructions>", enhanced_prompt)

    @patch('core.requests.get')
    def test_get_ollama_models(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_get.return_value = mock_response
        models = get_ollama_models()
        self.assertEqual(models, ["llama3:latest"])

    @patch('core.call_ollama')
    def test_choose_enhancement_strategy(self, mock_call_ollama):
        mock_call_ollama.return_value = "xml_structure"
        prompt = "test prompt"
        strategy = choose_enhancement_strategy(prompt, "llama3")
        self.assertEqual(strategy, "xml_structure")

if __name__ == '__main__':
    unittest.main()