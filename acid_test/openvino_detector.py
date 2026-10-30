"""Adapter for the user-selected INT8 OBB export as acid-on-stone detector."""
from pathlib import Path
import hashlib
import threading
import time
import xml.etree.ElementTree as ET

from acidphintegration.openvino_obb import OpenVINOOBB
from .onnx_detector import Detection


class OpenVINOAcidDetector:
    def __init__(self,path):
        self.path=Path(path)
        binary=self.path.with_suffix('.bin')
        for required in (self.path,binary):
            if not required.is_file():raise FileNotFoundError(required)
        tree=ET.parse(self.path)
        self.quantized=any(layer.get('type')=='FakeQuantize' for layer in tree.iter('layer'))
        if not self.quantized:raise ValueError('Expected the selected INT8 acid export')
        self.model=OpenVINOOBB(self.path)
        if (self.model.height,self.model.width)!=(640,640) or len(self.model.names)!=1:
            raise ValueError('Expected a single-class 640x640 OBB acid export')
        self.names=self.model.names
        self.lock=threading.Lock()
        digest=hashlib.sha256()
        for artifact in (self.path,binary):
            with artifact.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
        self.model_sha256=digest.hexdigest()

    def predict(self,frame,confidence):
        started=time.perf_counter()
        # One cached CPU session is shared by live and recorded recovery work.
        with self.lock:
            found=self.model.detect(frame,class_id=0,confidence=confidence)
        detections=[Detection(tuple(map(int,item['bbox'])),float(item['score'])) for item in found]
        return detections,(time.perf_counter()-started)*1000
