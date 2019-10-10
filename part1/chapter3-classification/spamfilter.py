import os
import tarfile
from six.moves import urllib
import email.policy
from email.parser import BytesParser
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np

from utils import EmailToWordCounterTransformer, WordCounterToVectorTransformer

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):  # we make th directory to download data into if we do not have the directory
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):  # downloading and extracting ham and spam files
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


# function to parse an email
def parse_email(filename, spam_path=SPAM_PATH, directory="spam"):
	with open(os.path.join(spam_path, directory, filename), 'rb') as file:
		return BytesParser(policy=email.policy.default).parse(file)

fetch_spam_data()

# loading spam and ham emails
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]

# using the filenames to load the emails
spam_emails = [parse_email(filename) for filename in spam_filenames]
ham_emails = [parse_email(filename, directory="easy_ham") for filename in ham_filenames]

# splitting to training and testing dataset
X = np.array(spam_emails + ham_emails)
y = np.array([1] * len(spam_emails) + [0] * len(ham_emails))  # label for spam is '1' and '0' for non-spam

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

preprocess_pipeline = Pipeline([
	("email_to_wordcount", EmailToWordCounterTransformer()),
	("wordcount_to_vector", WordCounterToVectorTransformer())])

X_train_trandformed = preprocess_pipeline.fit_transform(X_train)
