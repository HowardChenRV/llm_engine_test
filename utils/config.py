import configparser

class Config:
    def __init__(self, config_file="conf/config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        
    def get_log_level(self):
        """Get the log level from the configuration."""
        return self.config.get("log", "level", fallback="INFO")  # Default to WARNING if not set


# Example usage
# if __name__ == "__main__":
#     config = Config()
#     log_level = config.get_log_level()
#     print(f"Log level: {log_level}")