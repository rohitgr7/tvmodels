import os
import requests
import urllib.parse as urlparse
import torch

__all__ = ['load_pretrained']


def _download_file_from_google_drive(fid, dest):
    def _get_confirm_token(res):
        for k, v in res.cookies.items():
            if k.startswith('download_warning'):
                return v
        return None

    def _save_response_content(res, dest):
        CHUNK_SIZE = 32768
        with open(dest, "wb") as f:
            for chunk in res.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    res = sess.get(URL, params={'id': fid}, stream=True)
    token = _get_confirm_token(res)

    if token:
        params = {'id': fid, 'confirm': token}
        res = sess.get(URL, params=params, stream=True)
    _save_response_content(res, dest)


def _load_url(url, dest):
    if os.path.isfile(dest) and os.path.exists(dest):
        return dest
    print('[INFO]: Downloading weights...')
    fid = urlparse.parse_qs(urlparse.urlparse(url).query)['id'][0]
    _download_file_from_google_drive(fid, dest)
    return dest


def load_pretrained(m, meta, dest, pretrained=False):
    if pretrained:
        if len(meta) == 0:
            print('[INFO]: Pretrained model not available')
            return m
        if dest is None:
            dest = meta[0]
        m.load_state_dict(torch.load(_load_url(meta[1], dest)))
    return m
