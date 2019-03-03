# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the OpenNLP POS-tagger
#
# Copyright (C) Paulius Danenas
# Author: Paulius Danenas <danpaulius@gmail.com>

"""
A Python module for interfacing with the Apache OpenNLP package
"""

import os
import sys
import re
import gc
from subprocess import Popen, PIPE
from nltk.internals import find_binary
from nltk.tag.api import TaggerI

_opennlp_languages = ['da', 'de', 'en', 'es', 'nl', 'pt', 'se']

class OpenNLPTagger(TaggerI):

    def __init__(self, path_to_bin=None, path_to_model=None, language='en', verbose=False):
        """
        Initialize the OpenNLPTagger.

        :param path_to_bin: Path to bin directory of OpenNLP installation
        :param path_to_model: The path to OpenNLP POS tagger .bin file.
        :param language: Language to use; default setting is 'en'.

        """
        if path_to_model is None:
            raise LookupError('OpenNLP model file is not set!')
        self._model_path = path_to_model

        opennlp_paths = ['.', '/usr/bin', '/usr/local/apache-opennlp', '/opt/local/apache-opennlp', '~/apache-opennlp']
        opennlp_paths = list(map(os.path.expanduser, opennlp_paths))

        if language in _opennlp_languages:
            opennlp_bin_name = "opennlp"
            if sys.platform.startswith("win"):
                opennlp_bin_name += ".bat"
        else:
            raise LookupError('Language not in language list!')
        try:
            self._opennlp_bin = find_binary(opennlp_bin_name, os.path.join(path_to_bin, opennlp_bin_name),
                                          env_vars=('OPENNLP_HOME', 'OPENNLP'),
                                          searchpath=opennlp_paths, verbose=verbose)
        except LookupError:
            print('Unable to find the Apache OpenNLP run file!')


    def tag(self, sentences):

        if isinstance(sentences, list):
            _input = ''
            for sent in sentences:
                if isinstance(sent, list):
                    _input += ' '.join((x for x in sent))
                else:
                    _input += ' ' + sent
            _input = _input.lstrip()
            _input += '\n'
        else:
            _input = sentences

        # Run the tagger and get the output
        gc.collect()
        p = Popen([self._opennlp_bin, "POSTagger", self._model_path],
                  shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        (stdout, stderr) = p.communicate(_input)

        # Check the return code.
        if p.returncode != 0:
            raise OSError('OpenNLP command failed!')

        # Clean the execution time information
        output = re.sub(r"\nExecution time:(.*)$", "", stdout)
        # Output the tagged sentences
        tagged_tokens = []
        for tagged_word in output.strip().split(' '):
            words = tagged_word.split('_')
            tagged_tokens.append(('_'.join(words[:-1]), words[-1]))
        return tagged_tokens


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
