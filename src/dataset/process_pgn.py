import chess.pgn
import io
import json
import multiprocessing
import zstandard as zstd
from tqdm import tqdm


def process_game(game_pgn):
    game = chess.pgn.read_game(io.StringIO(game_pgn))
    if not game or ("Variant" in game.headers and game.headers["Variant"] != "Standard"):
        return []

    results = []
    board = game.board()
    for node in game.mainline():
        move = node.move
        board.push(move)
        comment = node.comment
        if "[%eval" in comment:
            eval_parts = comment.split("[%eval ")
            if len(eval_parts) > 1:
                eval_text = eval_parts[1].split("]")[0]
                fen = board.fen()
                fen_no_counts = ' '.join(fen.split(' ')[:-2])
                # fen_hash = hash(fen_no_counts)
                # fen_hash = fen_no_counts

                data_dict = {"FEN": fen, "FEN_processed": fen_no_counts}
                if eval_text.startswith("#"):
                    mate_in = int(eval_text.strip("#"))
                    data_dict["mate"] = mate_in
                else:
                    try:
                        eval_float = float(eval_text)
                        data_dict["cp"] = round(eval_float * 100)
                    except ValueError:
                        continue  # Skip if eval_text is not a number or mate notation
                results.append(data_dict)
        elif board.is_stalemate() or board.is_insufficient_material():
            fen = board.fen()
            fen_no_counts = ' '.join(fen.split(' ')[:-2])
            data_dict = {"FEN": fen, "FEN_processed": fen_no_counts, "cp": 0}
            results.append(data_dict)
        elif board.is_checkmate():
            # if it's white's turn then black checkmated white, sloppy but give it same value as checkmate in 1
            mate = -1 if board.turn == chess.WHITE else 1
            fen = board.fen()
            fen_no_counts = ' '.join(fen.split(' ')[:-2])
            data_dict = {"FEN": fen, "FEN_processed": fen_no_counts, "mate": mate}
            results.append(data_dict)
        else:
            break
    return results


def extract_fens_and_evals_to_jsonl(pgn_zst_path, output_path):
    seen_fens = set()

    with open(pgn_zst_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader, open(output_path, "a") as output_file:
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')
            pbar = tqdm(desc="Extracting games", unit="games")

            games_batch = []
            batch_size = 20000
            game_pgn = ""

            for line in text_reader:
                if line.strip() == "" and game_pgn:
                    games_batch.append(game_pgn)
                    game_pgn = ""  # Reset for the next game
                    if len(games_batch) >= batch_size:
                        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool:
                            batch_results = pool.map(process_game, games_batch)
                        for game_results in batch_results:
                            for result in game_results:
                                fen_hash = hash(result["FEN_processed"])
                                if fen_hash not in seen_fens:
                                    seen_fens.add(fen_hash)
                                    del result["FEN_processed"]
                                    json.dump(result, output_file)
                                    output_file.write('\n')
                        games_batch = []  # Clear the batch after processing
                        pbar.update(batch_size)
                else:
                    game_pgn += line

            # Process any remaining games in the last batch
            if games_batch:
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool:
                    batch_results = pool.map(process_game, games_batch)
                for game_results in batch_results:
                    for result in game_results:
                        fen_hash = hash(result["FEN_processed"])
                        if fen_hash not in seen_fens:
                            seen_fens.add(fen_hash)
                            del result["FEN_processed"]
                            json.dump(result, output_file)
                            output_file.write('\n')
                pbar.update(len(games_batch))

            pbar.close()


# def extract_fens_and_evals_to_jsonl(pgn_zst_path, output_path):
#     seen_fens = set()
#
#     with open(pgn_zst_path, 'rb') as compressed_file:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(compressed_file) as reader, open(output_path, "w") as output_file:
#             text_reader = io.TextIOWrapper(reader, encoding='utf-8')
#             pbar = tqdm(desc="Extracting games", unit=" games")
#             while True:
#                 game = chess.pgn.read_game(text_reader)
#                 if game is None:o
#                     break  # End of the PGN file
#
#                 pbar.update(1)  # Manually update tqdm for each game processed
#
#                 if "Variant" in game.headers and game.headers["Variant"] != "Standard":
#                     continue  # Skip variant games
#
#                 board = game.board()
#                 for node in game.mainline():
#                     move = node.move
#                     board.push(move)
#                     comment = node.comment
#                     # Look specifically for the [%eval ...] part in the comment
#                     if "[%eval" in comment:
#                         eval_parts = comment.split("[%eval ")
#                         if len(eval_parts) > 1:
#                             eval_text = eval_parts[1].split("]")[0]  # Isolate the evaluation text
#                             fen = board.fen()
#                             fen_no_counts = ' '.join(fen.split(' ')[:-2])
#                             fen_hash = hash(fen_no_counts)
#
#                             if fen_hash not in seen_fens:
#                                 seen_fens.add(fen_hash)
#                                 data_dict = {"FEN": fen}
#                                 if eval_text.startswith("#"):
#                                     # Mate situation
#                                     mate_in = int(eval_text.strip("#"))
#                                     data_dict["mate"] = mate_in
#                                 else:
#                                     try:
#                                         # Numeric evaluation
#                                         eval_float = float(eval_text)
#                                         data_dict["cp"] = round(eval_float * 100)
#                                     except ValueError:
#                                         continue  # Ignore if evaluation is not float or mate notation
#
#                                 json.dump(data_dict, output_file)
#                                 output_file.write('\n')  # New line for the next JSON object
#                     else:
#                         break
#
#                 pbar.set_description(f"Processed {pbar.n} games")
#                 pbar.refresh()  # Update the progress bar display
#
#             pbar.close()  # Ensure the progress bar closes properly


if __name__ == '__main__':
    # 2017-05 doesn't have stalemates, insufficient material draws, checkmates
    # 2017-06 processed to include stalemates, insufficient material draws, checkmates
    # (checkmates are included as mate in +/-1)
    extract_fens_and_evals_to_jsonl(
        # r"C:\Users\Ahmad-personal\Downloads\lichess_db_standard_rated_2024-03.pgn.zst",
        r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\data\lichess_db_standard_rated_2025-11.pgn.zst",
        r"C:\Users\Ahmad-personal\PycharmProjects\chess_stockfish_evals_v2\data\lichess_db_standard_rated_2025-11.jsonl",
    )

