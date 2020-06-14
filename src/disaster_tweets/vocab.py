
DEFAULT_UNKNOWN_TOKEN = "<UNK>"


class Vocabulary(object):
    def __init__(self, token_map=None, add_unknown=True, unknown_token=DEFAULT_UNKNOWN_TOKEN) -> None:
        if token_map is None:
            token_map = {}
        self.token_map = token_map
        self.index_map = {idx: token for token, idx in self.token_map.items()}

        self.add_unknown = add_unknown
        self.unknown_token = unknown_token

        self.unknown_index = -1
        if add_unknown:
            self.unknown_index = self.add(unknown_token)

    def add(self, token: str) -> int:
        if token not in self.token_map:
            index = len(self)
            self.token_map[token] = index
            self.index_map[index] = token
        return self.lookup(token)

    def lookup(self, token: str) -> int:
        if self.add_unknown:
            return self.token_map.get(token, self.unknown_token)
        return self.token_map[token]

    def at(self, index: int) -> str:
        return self.index_map[index]

    def __str__(self) -> str:
        return f"<Vocabulary(len={len(self)})>"

    def __len__(self) -> int:
        return len(self.token_map)
        