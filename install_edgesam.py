"""Download the upstream EdgeSAM ONNX pair; never runs inside the tracker."""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '32266dda3cf1d5ea1cdccd386de1eb3355d0f0b9'
BASE = 'https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/'+REVISION+'/'


def main():
    directory = Path(__file__).resolve().parent/'weights/edgesam'
    directory.mkdir(parents=True,exist_ok=True)
    manifest = dict(repository='chongzhou/EdgeSAM', revision=REVISION, files={})
    for name in ('edge_sam_3x_encoder.onnx','edge_sam_3x_decoder.onnx'):
        target = directory/name
        metadata = json.load(urllib.request.urlopen(
            'https://huggingface.co/api/spaces/chongzhou/EdgeSAM/tree/'+REVISION+'/weights',timeout=30))
        info = next(item for item in metadata if item['path']=='weights/'+name)
        expected = info['lfs']['oid']
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            temp=target.with_suffix('.downloading')
            try:
                with urllib.request.urlopen(BASE+'weights/'+name,timeout=90) as response,temp.open('wb') as output:
                    while chunk:=response.read(1024*1024):output.write(chunk)
                if hashlib.sha256(temp.read_bytes()).hexdigest()!=expected:
                    raise OSError('Model checksum mismatch: '+name)
                temp.replace(target)
            finally:temp.unlink(missing_ok=True)
        manifest['files'][name]=dict(sha256=expected,bytes=target.stat().st_size)
        print(name,manifest['files'][name],flush=True)
    (directory/'models.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
