from typing import List

from recordclass import RecordClass


class RevisionIds(RecordClass):

    revision: str = None
    method_ids: List[int] = None

    def init(self):
        self.method_ids = []
