import os

os.environ['JDK_HOME'] = "C:\Program Files\Java\jdk-11.0.10"
os.environ['JAVA_HOME'] = "C:\Program Files\Java\jdk-11.0.10"

os.environ['PATH'] += ';C:\\Program Files\\Java\\jdk-11.0.10\\jre\\bin\\server\\'


from pyserini.search import SimpleSearcher

SimpleSearcher.list_prebuilt_indexes()