
def remove_full_half_moves_from_fen(fen: str) -> str:
    parts = fen.split(' ')
    if len(parts) == 6:
        return ' '.join(parts[:-2])
    return fen


def expand_fen_string(fen: str) -> str:
    # Split the FEN string into its components
    parts = fen.split(' ')
    board, rest = parts[0], parts[1:]

    # Expand the first component
    expanded_board = ""
    for char in board:
        if char.isdigit():
            expanded_board += '.' * int(char)
        else:
            expanded_board += char

    # Reassemble the FEN string
    expanded_fen = ' '.join([expanded_board] + rest)
    return remove_full_half_moves_from_fen(expanded_fen)