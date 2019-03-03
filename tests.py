# -*- coding: utf-8 -*-
import os
import unittest

from nltk_opennlp.chunkers import OpenNLPChunker, OpenNERChunker, OpenNERChunkerMulti
from nltk_opennlp.taggers import OpenNLPTagger

opennlp_dir = 'apache-opennlp'    # Path to Apache OpenNLP
models_dir = 'opennlp_models'     # Path to OpenNLP models directory

class OpenNLPTest(unittest.TestCase):

    def test_opennlp_tagger(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = 'Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .'
        en_tags = tt.tag(phrase)
        print(en_tags)
        assert en_tags[0][0] == 'Pierre'
        assert en_tags[0][1] == 'NNP'

    def test_opennlp_tagger_list(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = ['Pierre', 'Vinken' ',' '61', 'years', 'old', ',', 'will', 'join',
                  'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.']
        en_tags = tt.tag(phrase)
        print(en_tags)
        assert en_tags[0][0] == 'Pierre'
        assert en_tags[0][1] == 'NNP'


    def test_opennlp_chunker(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = 'Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .'
        sentence = tt.tag(phrase)
        cp = OpenNLPChunker(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                            path_to_chunker=os.path.join(models_dir, 'en-chunker.bin'))
        print(cp.parse(sentence))


    def test_opennlp_chunker_de(self):
        language = 'de'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'de-pos-maxent.bin'))
        phrase = 'Das Haus hat einen großen hübschen Garten.'
        sentence = tt.tag(phrase)
        print(sentence)
        # There should not be OpenNLP chunker for German language, thus OSError is thrown in Linux
        if os.name != 'nt':
            with self.assertRaises(OSError):
                cp = OpenNLPChunker(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                                    path_to_chunker=os.path.join(models_dir, 'de-chunker.bin'))
                print(cp.parse(sentence))


    def test_opennlp_ner_chunker(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = 'Pierre Vinken , 61 years old , will join Martin Vinken as a nonexecutive director Nov. 29 .'
        sentence = tt.tag(phrase)
        cp = OpenNERChunker(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                            path_to_chunker=os.path.join(models_dir,
                                                         '{}-chunker.bin'.format(language)),
                            path_to_ner_model=os.path.join(models_dir,
                                                           '{}-ner-person.bin'.format(language)))
        print(cp.parse(sentence))


    def test_opennlp_ner_chunker_bracketed(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = 'Pierre Vinken , ( 61 years old ) , will join Martin Vinken as a nonexecutive director Nov. 29 .'
        sentence = tt.tag(phrase)
        cp = OpenNERChunker(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                            path_to_chunker=os.path.join(models_dir,
                                                         '{}-chunker.bin'.format(language)),
                            path_to_ner_model=os.path.join(models_dir,
                                                           '{}-ner-person.bin'.format(language)))
        print(cp.parse(sentence))


    def test_opennlp_ner_chunker_with_punc(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = 'Pierre Vinken , 61 years old , will join Martin Vinken as a nonexecutive director Nov. 29 .'
        sentence = tt.tag(phrase)
        cp = OpenNERChunker(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                            path_to_chunker=os.path.join(models_dir,
                                                         '{}-chunker.bin'.format(language)),
                            path_to_ner_model=os.path.join(models_dir,
                                                           '{}-ner-person.bin'.format(language)),
                            use_punc_tag=True)
        print(cp.parse(sentence))


    def test_opennlp_ner_multichunker(self):
        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = 'John Haddock , 32 years old male , travelled to Cambridge , USA in October 20 while paying 6.50 dollars for the ticket'
        sentence = tt.tag(phrase)
        cp = OpenNERChunkerMulti(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                                 path_to_chunker=os.path.join(models_dir,
                                                              '{}-chunker.bin'.format(language)),
                                 ner_models=[
                                     os.path.join(models_dir, '{}-ner-person.bin'.format(language)),
                                     os.path.join(models_dir, '{}-ner-date.bin'.format(language)),
                                     os.path.join(models_dir, '{}-ner-location.bin'.format(language)),
                                     os.path.join(models_dir, '{}-ner-time.bin'.format(language)),
                                     os.path.join(models_dir, '{}-ner-money.bin'.format(language))])
        print(cp.parse(sentence))


if __name__ == '__main__':
    unittest.main()
