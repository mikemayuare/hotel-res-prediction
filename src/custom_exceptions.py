import sys


class CustomException(Exception):
    """
    A custom exception that automatically includes the file name
    and line number in the error message.
    """

    def __init__(self, error_message):
        super().__init__(error_message)

        self.error_message = self.get_detailed_error_message(error_message)

    @staticmethod
    def get_detailed_error_message(error_message):
        _, _, exc_tb = sys.exc_info()

        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: {error_message}"

        return error_message

    def __str__(self):
        return self.error_message
