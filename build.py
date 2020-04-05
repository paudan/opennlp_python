import os, shutil, sys, errno
import tarfile
from zipfile import ZipFile
if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
    from urllib.error import HTTPError
else:
    import urllib2
    import urlparse
    import urllib2.HTTPError as HTTPError
from pybuilder.core import init, task, Author


name = 'opennlp-sandbox'
authors = [Author('Paulius Danenas', 'danpaulius@gmail.com')]
license = 'Apache License, Version 2.0'
summary = 'Setup OpenNLP environment'
version = '1.0'
default_task = ['setup_opennlp']

# OpenNLP settings
install_dir = os.getcwd()
languages = ['en','de']
OPENNLP_VER = '1.8.4'
OPENNLP_DIR = 'apache-opennlp'
OPENNLP_MODELS_DIR = 'opennlp_models'


# Adopted from https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
def download_file(url, desc=None):
    u = urllib2.urlopen(url)
    scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
    filename = os.path.basename(path)
    if not filename:
        filename = 'downloaded.file'
    if desc:
        filename = os.path.join(desc, filename)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)
            print(status, end="")
        print()

    return filename

def get_file(source, url, targetDir, logger):
    if not targetDir is None:
        source = os.path.join(targetDir, source)
    if not os.path.isfile(source):
        try:
            download_file(url, desc=targetDir)
        except HTTPError as e:
            logger.error("Error downloading file from '{}': {}".format(url, e.reason))


def extract_file(source, targetdir):
    topdir = targetdir
    if source.endswith(".tar.gz") or source.endswith(".tar.bz2"):
        tar = tarfile.open(source)
        tar.extractall(targetdir)
        topdir = os.path.commonprefix(tar.getnames()).rstrip("/")
        tar.close()
    elif source.endswith(".zip"):
        zip = ZipFile(source, 'r')
        zip.extractall(targetdir)
        topdir = os.path.commonprefix(zip.namelist()).rstrip("/")
        zip.close()
    return topdir if targetdir is None else targetdir


def copy_source(source, target, logger, isTargetDir=False):

    def copytree(src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    if isTargetDir and not (os.path.exists(target) and os.path.isdir(target)):
        os.mkdir(target)
    try:
        copytree(source, target)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy(source, target)
        else:
            logger.error(exc.strerror)


def download_opennlp_files(langlist, logger):
    opennlp_url = 'http://opennlp.sourceforge.net/models-1.5'
    lang_map = {
        'da': ['da-token.bin', 'da-sent.bin', 'da-pos-maxent.bin', 'da-pos-perceptron.bin'],
        'de': ['de-token.bin', 'de-sent.bin', 'de-pos-maxent.bin', 'de-pos-perceptron.bin'],
        'en': ['en-token.bin', 'en-sent.bin', 'en-pos-maxent.bin', 'en-pos-perceptron.bin',
               'en-ner-date.bin', 'en-ner-location.bin', 'en-ner-money.bin', 'en-ner-organization.bin',
               'en-ner-percentage.bin', 'en-ner-person.bin', 'en-ner-time.bin', 'en-chunker.bin',
               'en-parser-chunking.bin'],
        'es': ['es-ner-person.bin', 'es-ner-organization.bin', 'es-ner-location.bin', 'es-ner-misc.bin'],
        'nl': ['nl-token.bin', 'nl-sent.bin', 'nl-pos-maxent.bin', 'nl-pos-perceptron.bin',
               'nl-ner-person.bin', 'nl-ner-location.bin', 'nl-ner-misc.bin', 'nl-ner-organization.bin'],
        'pt': ['pt-token.bin', 'pt-sent.bin', 'pt-pos-maxent.bin', 'pt-pos-perceptron.bin'],
        'se': ['se-token.bin', 'se-sent.bin', 'se-pos-maxent.bin', 'se-pos-perceptron.bin'],
    }
    for lang in langlist:
        for file in lang_map.get(lang, []):
            if not os.path.isfile(os.path.join(OPENNLP_MODELS_DIR, file)):
                try:
                    download_file("{0}/{1}".format(opennlp_url, file), desc=OPENNLP_MODELS_DIR)
                except HTTPError as exc:
                    logger.error(exc.reason)


# @init
# def initialize(project):
#     project.depends_on_requirements("requirements.txt")
#     global install_dir
#     install_dir = project.get_property('install_dir')
#     if install_dir is None:
#         project.set_property('install_dir', os.getcwd())
#     if not (os.path.exists(install_dir) and os.path.isdir(install_dir)):
#         os.mkdir(install_dir)
#     install_dir = os.path.realpath(install_dir)
#     os.chdir(install_dir)


@task
def setup_opennlp(logger):
    opennlp_file = f'https://archive.apache.org/dist/opennlp/opennlp-{OPENNLP_VER}/apache-opennlp-{OPENNLP_VER}-bin.zip'
    logger.info('Downloading OpenNLP')
    if os.path.exists(OPENNLP_DIR) and os.path.isdir(OPENNLP_DIR):
        shutil.rmtree(OPENNLP_DIR)
    opennlp_filename = opennlp_file.split("/")[-1]
    get_file(opennlp_filename, opennlp_file, None, logger)
    extracted_dir = extract_file(opennlp_filename, None)
    os.remove(opennlp_filename)
    os.rename(extracted_dir, OPENNLP_DIR)
    for root, dirs, files in os.walk(os.path.join(OPENNLP_DIR, 'bin')):
        for file in files:
            os.chmod(os.path.join(root, file), 0o777)
    logger.info('Downloading OpenNLP models for selected languages')
    if not (os.path.exists(OPENNLP_MODELS_DIR) and os.path.isdir(OPENNLP_MODELS_DIR)):
        os.mkdir(OPENNLP_MODELS_DIR)
    download_opennlp_files(languages, logger)
    logger.info("Done")
