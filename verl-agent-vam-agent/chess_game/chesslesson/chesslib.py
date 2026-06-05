"""Minimal but correct chess engine, ported 1:1 from chesslib.js.

FEN parsing, pseudo/legal move generation, move application (castling, en
passant, promotion) and check/checkmate/stalemate detection. Pure Python.
Board is board[0..7][0..7]; row 0 = rank 8, col 0 = file a. Pieces are FEN
letters (uppercase = white) or None.
"""
from copy import deepcopy


def sq_to_rc(name):          # 'e2' -> (6, 4)
    return (8 - int(name[1]), ord(name[0]) - 97)


def rc_to_sq(r, c):
    return chr(97 + c) + str(8 - r)


def is_w(p):
    return p is not None and p == p.upper()


def color_of(p):
    return 'w' if is_w(p) else 'b'


def in_board(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def parse(fen):
    parts = fen.strip().split()
    placement = parts[0]
    turn = parts[1] if len(parts) > 1 else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    ep = parts[3] if len(parts) > 3 else '-'
    board = []
    for row in placement.split('/'):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend([None] * int(ch))
            else:
                cells.append(ch)
        board.append(cells)
    return {
        'board': board,
        'turn': turn,
        'castling': {'K': 'K' in castling, 'Q': 'Q' in castling,
                     'k': 'k' in castling, 'q': 'q' in castling},
        'ep': None if ep == '-' else ep,
    }


def clone(s):
    return {'board': [row[:] for row in s['board']], 'turn': s['turn'],
            'castling': dict(s['castling']), 'ep': s['ep']}


KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BISHOP = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def attacked(s, r, c, by):
    """Is square (r,c) attacked by color `by` ('w'/'b')?"""
    b = s['board']
    pr = r + 1 if by == 'w' else r - 1
    for dc in (-1, 1):
        if in_board(pr, c + dc):
            p = b[pr][c + dc]
            if p and color_of(p) == by and p.upper() == 'P':
                return True
    for dr, dc in KNIGHT:
        rr, cc = r + dr, c + dc
        if in_board(rr, cc):
            p = b[rr][cc]
            if p and color_of(p) == by and p.upper() == 'N':
                return True
    for dr, dc in KING:
        rr, cc = r + dr, c + dc
        if in_board(rr, cc):
            p = b[rr][cc]
            if p and color_of(p) == by and p.upper() == 'K':
                return True
    for dr, dc in BISHOP:
        rr, cc = r + dr, c + dc
        while in_board(rr, cc):
            p = b[rr][cc]
            if p:
                if color_of(p) == by and p.upper() in ('B', 'Q'):
                    return True
                break
            rr += dr
            cc += dc
    for dr, dc in ROOK:
        rr, cc = r + dr, c + dc
        while in_board(rr, cc):
            p = b[rr][cc]
            if p:
                if color_of(p) == by and p.upper() in ('R', 'Q'):
                    return True
                break
            rr += dr
            cc += dc
    return False


def king_square(s, color):
    K = 'K' if color == 'w' else 'k'
    for r in range(8):
        for c in range(8):
            if s['board'][r][c] == K:
                return (r, c)
    return None


def in_check(s, color):
    k = king_square(s, color)
    return attacked(s, k[0], k[1], 'b' if color == 'w' else 'w') if k else False


def pseudo(s):
    """Pseudo-legal moves for side to move. Each: {from,to,promo?,flag?}."""
    b = s['board']
    side = s['turn']
    opp = 'b' if side == 'w' else 'w'
    moves = []

    def add(fr, fc, tr, tc, extra=None):
        m = {'from': rc_to_sq(fr, fc), 'to': rc_to_sq(tr, tc)}
        if extra:
            m.update(extra)
        moves.append(m)

    for r in range(8):
        for c in range(8):
            p = b[r][c]
            if not p or color_of(p) != side:
                continue
            t = p.upper()
            if t == 'P':
                direction = -1 if side == 'w' else 1
                start = 6 if side == 'w' else 1
                last = 0 if side == 'w' else 7
                r1 = r + direction
                if in_board(r1, c) and not b[r1][c]:
                    if r1 == last:
                        for pr in ('q', 'r', 'b', 'n'):
                            add(r, c, r1, c, {'promo': pr})
                    else:
                        add(r, c, r1, c)
                        if r == start and not b[r + 2 * direction][c]:
                            add(r, c, r + 2 * direction, c, {'flag': 'double'})
                for dc in (-1, 1):
                    cc = c + dc
                    if not in_board(r1, cc):
                        continue
                    tp = b[r1][cc]
                    if tp and color_of(tp) == opp:
                        if r1 == last:
                            for pr in ('q', 'r', 'b', 'n'):
                                add(r, c, r1, cc, {'promo': pr})
                        else:
                            add(r, c, r1, cc)
                    elif not tp and s['ep'] and rc_to_sq(r1, cc) == s['ep']:
                        add(r, c, r1, cc, {'flag': 'ep'})
            elif t == 'N':
                for dr, dc in KNIGHT:
                    rr, cc = r + dr, c + dc
                    if in_board(rr, cc) and (not b[rr][cc] or color_of(b[rr][cc]) == opp):
                        add(r, c, rr, cc)
            elif t == 'K':
                for dr, dc in KING:
                    rr, cc = r + dr, c + dc
                    if in_board(rr, cc) and (not b[rr][cc] or color_of(b[rr][cc]) == opp):
                        add(r, c, rr, cc)
                rights = s['castling']
                home_r = 7 if side == 'w' else 0
                if r == home_r and c == 4 and not in_check(s, side):
                    ks = rights['K'] if side == 'w' else rights['k']
                    qs = rights['Q'] if side == 'w' else rights['q']
                    if (ks and not b[home_r][5] and not b[home_r][6]
                            and b[home_r][7] and b[home_r][7].upper() == 'R'
                            and not attacked(s, home_r, 5, opp) and not attacked(s, home_r, 6, opp)):
                        add(r, c, home_r, 6, {'flag': 'O-O'})
                    if (qs and not b[home_r][3] and not b[home_r][2] and not b[home_r][1]
                            and b[home_r][0] and b[home_r][0].upper() == 'R'
                            and not attacked(s, home_r, 3, opp) and not attacked(s, home_r, 2, opp)):
                        add(r, c, home_r, 2, {'flag': 'O-O-O'})
            else:
                dirs = BISHOP if t == 'B' else ROOK if t == 'R' else KING
                for dr, dc in dirs:
                    rr, cc = r + dr, c + dc
                    while in_board(rr, cc):
                        tp = b[rr][cc]
                        if not tp:
                            add(r, c, rr, cc)
                        else:
                            if color_of(tp) == opp:
                                add(r, c, rr, cc)
                            break
                        rr += dr
                        cc += dc
    return moves


def apply_move(s, m):
    """Apply a fully-specified pseudo move; return new state (no legality check)."""
    ns = clone(s)
    b = ns['board']
    fr, fc = sq_to_rc(m['from'])
    tr, tc = sq_to_rc(m['to'])
    p = b[fr][fc]
    side = s['turn']
    home_r = 7 if side == 'w' else 0
    if m.get('flag') == 'ep':
        b[fr][tc] = None
    b[tr][tc] = p
    b[fr][fc] = None
    if m.get('promo'):
        b[tr][tc] = m['promo'].upper() if side == 'w' else m['promo'].lower()
    if m.get('flag') == 'O-O':
        b[home_r][5] = b[home_r][7]
        b[home_r][7] = None
    if m.get('flag') == 'O-O-O':
        b[home_r][3] = b[home_r][0]
        b[home_r][0] = None
    cr = ns['castling']
    if p.upper() == 'K':
        if side == 'w':
            cr['K'] = cr['Q'] = False
        else:
            cr['k'] = cr['q'] = False

    def touch(sq):
        if sq == 'a1':
            cr['Q'] = False
        if sq == 'h1':
            cr['K'] = False
        if sq == 'a8':
            cr['q'] = False
        if sq == 'h8':
            cr['k'] = False
    touch(m['from'])
    touch(m['to'])
    ns['ep'] = rc_to_sq((fr + tr) // 2, fc) if m.get('flag') == 'double' else None
    ns['turn'] = 'b' if side == 'w' else 'w'
    return ns


def legal_moves(s):
    side = s['turn']
    return [m for m in pseudo(s) if not in_check(apply_move(s, m), side)]


def find_move(s, uci):
    import re
    uci = re.sub(r'[^a-h1-8qrbn]', '', uci.strip().lower())
    if len(uci) < 4:
        return None
    frm, to = uci[0:2], uci[2:4]
    promo = uci[4] if len(uci) > 4 else None
    cands = [m for m in legal_moves(s) if m['from'] == frm and m['to'] == to]
    if not cands:
        return None
    if promo:
        return next((m for m in cands if m.get('promo') == promo), None)
    return next((m for m in cands if m.get('promo') == 'q'), cands[0])


def board_fen(board):
    rows = []
    for row in board:
        o, e = '', 0
        for p in row:
            if not p:
                e += 1
            else:
                if e:
                    o += str(e)
                    e = 0
                o += p
        if e:
            o += str(e)
        rows.append(o)
    return '/'.join(rows)
