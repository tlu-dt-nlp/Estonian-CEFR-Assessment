import itertools
import unittest
from pprint import pprint
from typing import List
from dataclasses import asdict

from gec_worker.utils import generate_spans
from gec_worker.dataclasses import Correction


def apply_corrections(original: str, corrections: List[Correction]):
    idx = 0
    text = ''
    for correction in corrections:
        if correction.replacements[0].value is None:
            value = ''
        else:
            value = correction.replacements[0].value
        text += original[idx:correction.span.start] + value
        idx = correction.span.end
    text += original[idx:]

    return text


class Test(unittest.TestCase):
    def test_GEC_nelb_ut_model(self):
        """
        Test that the model can be loaded and used.
        :return:
        """
        from gec_worker import GEC, read_gec_config
        from gec_worker.dataclasses import Response, Request

        gec_config = read_gec_config('models/GEC-nelb-1.3b.yaml')
        gec = GEC(gec_config)
        request = Request(text="Juku joksis koolis. Aitüma.", language='et')
        response = gec.process_request(request)
        pprint(asdict(response))
        self.assertIsInstance(response, Response)

    def test_GEC_synthetic_pretrain_ut_model(self):
        """
        Test that the model can be loaded and used.
        :return:
        """
        from gec_worker import GEC, read_gec_config
        from gec_worker.dataclasses import Response, Request

        gec_config = read_gec_config('models/GEC-synthetic-pretrain-ut-ft.yaml')
        gec = GEC(gec_config)
        request = Request(text="Juku joksis koolis. Aitüma.", language='et')
        response = gec.process_request(request)
        pprint(asdict(response))
        self.assertIsInstance(response, Response)

    def test_GEC_noisy_nmt_ut_model(self):
        """
        Test that the model can be loaded and used.
        :return:
        """
        from gec_worker import GEC, read_gec_config
        from gec_worker.dataclasses import Response, Request

        gec_config = read_gec_config('models/GEC-noisy-nmt-ut.yaml')
        gec = GEC(gec_config)
        request = Request(text="Juku joksis koolis. Aitüma", language='et')
        response = gec.process_request(request)
        pprint(asdict(response))
        self.assertIsInstance(response, Response)

    def test_spell_model(self):
        """
        Test that the model can be loaded and used.
        :return:
        """
        from gec_worker import Speller, read_speller_config
        from gec_worker.dataclasses import Response, Request

        # speller_config = read_spell_model_config("models/spell_etnc19_web_2019.yaml")
        speller_config = read_speller_config("models/spell_etnc19_reference_corpus_6000000_web_2019_600000.yaml")
        # speller_config = read_spell_model_config("models/spell_etnc19_reference_corpus_model_6000000_lines.yaml")
        speller = Speller(speller_config)

        request = Request(text="Juku joksis koolis. Aitüma.", language='et')
        response = speller.process_request(request)
        pprint(asdict(response))
        self.assertIsInstance(response, Response)


def test_multiple_correction_model(self):
    """
    Test that the model can be loaded and used.
    :return:
    """
    from gec_worker import GEC, read_gec_config
    from gec_worker import Speller, read_speller_config

    from gec_worker.dataclasses import Response, Request
    from gec_worker import MultiCorrector

    gec_config = read_gec_config('models/GEC-noisy-nmt-ut.yaml')
    gec = GEC(gec_config)

    speller_config = read_speller_config("models/spell_etnc19_reference_corpus_6000000_web_2019_600000.yaml")
    speller = Speller(speller_config)

    multi_corrector = MultiCorrector()
    multi_corrector.add_corrector(speller)
    multi_corrector.add_corrector(gec)

    request = Request(text="Juku joksis koolis. Aitüma.", language='et')
    response = multi_corrector.process_request(request)

    pprint(asdict(response))
    self.assertIsInstance(response, Response)


def test_generate_spans(self):
    """
    Test that correct differences are detected and the sentence can be restored to the target.
    :return:
    """

    a = "Lapsed peavad näitama paremaid tulemusi, rahuldada vanemate, õetajate ja ühiskonna mis tõttu " \
        "nõyrdfujudmisi, kuid kõik see võib tekitada lapses  stressi."
    b = "Lapsed jah peavad näitama häid ka tulemusi, et vanemate, õpetajae ja ühiskonna nõudmisi, " \
        "kuiyhfd kõik see võib tekitada lapses stressi."

    # combining sentences to test differences in both directions
    delimiters = ['', '\n\n', '']
    source = ''.join(itertools.chain.from_iterable(zip(delimiters, [a, b]))) + delimiters[-1]
    target = ''.join(itertools.chain.from_iterable(zip(delimiters, [b, a]))) + delimiters[-1]

    corrections = generate_spans([a, b], [b, a], ['', '\n\n', ''])
    pprint(corrections)
    restored = apply_corrections(source, corrections)
    self.assertEqual(restored, target)


if __name__ == '__main__':
    unittest.main()
