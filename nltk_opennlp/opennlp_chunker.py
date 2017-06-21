# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the OpenNLP chunker
#
# Copyright (C) Paulius Danenas
# Author: Paulius Danenas <danpaulius@gmail.com>

"""
A Python module for interfacing with the Apache OpenNLP package and obtaining parse tree
"""

import os
import sys
import re
from subprocess import Popen, PIPE
from nltk.internals import find_binary
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree

_opennlp_languages = ['da', 'de', 'en', 'es', 'nl', 'pt', 'se']

class OpenNLPChunker(ChunkParserI):

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


    def parse(self, tokens):
        _input = ' '.join([token[0] + "_" + token[1] for token in tokens])

        p = Popen([self._opennlp_bin, "ChunkerME", self._model_path],
                  shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        if sys.version_info >= (3,):
            (stdout, stderr) = p.communicate(bytes(_input, 'UTF-8'))
            stdout = stdout.decode('utf-8')
        else:
            (stdout, stderr) = p.communicate(_input)

        # Check the return code.
        if p.returncode != 0:
            raise OSError('OpenNLP command failed!')

        # Clean the execution time information
        output = re.sub(r"\nExecution time:(.*)$", "", stdout)
        # Transform into compatible parse tree string
        output = output.replace("[", "(").replace("]", ")")
        pattern = re.compile(r'\s+([^_\(\)]+)_([^_\(\)]+)\s+')
        output = re.sub(pattern, r' (\1 \2) ', output)
        output = re.sub(pattern, r' (\1 \2) ', output)
        output = "(ROOT {} )".format(output)
        try:
            parse = Tree.fromstring(output)
        except Exception:
            parse = None
        return parse
