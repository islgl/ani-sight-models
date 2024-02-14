import json
from typing import Dict


class CustomResponse:
    """ Custom response

    Attributes:
        status: The status code of the response
        msg: The message of the response
        data: The data of the response, default is None
    """

    def __init__(self, status: str, msg: str, data: Dict = None):
        self.status = status
        self.msg = msg
        self.data = data

    def __repr__(self):
        return f"CustomResponse(status_code={self.status}, msg={self.msg}, data={self.data})"

    def to_json(self):
        response = {
            "status": self.status,
            "msg": self.msg
        }
        if self.data is not None:
            response["data"] = self.data
        return json.dumps(response)
