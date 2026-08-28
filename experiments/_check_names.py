"""模組層級的「用在定義之前」檢查。⛔ py_compile / ast.parse 都擋不住這種錯。

同一個形狀被咬過三次（handoff 2026-08-23 ⑨ 兩次；2026-08-24 一次：把 T_CAP 的夾子寫到
T_CAP_REQ 定義的【上面】，AST 過、job 送出去、在計算節點炸 NameError）。

範圍（⛔ 故意窄，因為窄才不會誤報）：
  ✔ 模組層級語句直接讀到的名字
  ✔ 複合語句（for/while/if/with）在【進入之前】就求值的部分：iter / test / context
  ✘ 複合語句與函式的 body —— 那些名字的先後在同一個語句內互相定義，
    逐句比對會冒出一大堆假陽性（本檔第一版就是這樣壞掉的）。
⇒ 抓得到今天這個 bug 的形狀，而且不吵。

用法：  python experiments/_check_names.py <檔案...>       （回傳碼 1 = 有問題）
        python experiments/_check_names.py --self-test     （驗這支自己還會不會叫）
"""
import ast
import builtins
import sys


def _bound(node):
    """這個語句（含其 body）會綁定哪些名字。"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.comprehension,)):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)   # ⚠️ lambda / def 的參數是 ast.arg，⛔ 不是 Name（漏掉會誤報成未定義）
    return out


def _eager_parts(stmt):
    """一個語句裡【在執行到它時就會求值】的子樹（⛔ 不含 body）。"""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parts = list(stmt.decorator_list) + list(stmt.args.defaults)
        parts += [d for d in stmt.args.kw_defaults if d]
        return parts
    if isinstance(stmt, ast.ClassDef):
        return list(stmt.decorator_list) + list(stmt.bases)
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, ast.While):
        return [stmt.test]
    if isinstance(stmt, ast.If):
        return [stmt.test]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [it.context_expr for it in stmt.items]
    if isinstance(stmt, ast.Try):
        return []
    return [stmt]


def check(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    defined = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__"}
    seen, problems = set(), []
    for stmt in tree.body:
        local = _bound(stmt) if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else set()
        for part in _eager_parts(stmt):
            for n in ast.walk(part):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    if n.id not in defined and n.id not in local and (n.lineno, n.id) not in seen:
                        seen.add((n.lineno, n.id))
                        problems.append((n.lineno, n.id))
                elif isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    defined |= _bound(n)   # 這些自己會綁名字，別把它們的變數當未定義
        defined |= _bound(stmt)
    return problems


SELF_TEST = '''
import os
A = int(os.environ.get("X", 1))
B = min(C, A)          # ← 這一行要被抓到：C 還沒定義
C = 5
for i in zip(D):       # ← 這一行也要被抓到：D 還沒定義
    q = i
D = [1]
mse = lambda p, a: p - a   # ⛔ 不可以被抓到：p / a 是 lambda 的參數（第二版誤報過）
E = sorted([z for z in D])  # ⛔ 也不可以：z 是 comprehension 的變數
'''

if sys.argv[1:2] == ["--self-test"]:
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SELF_TEST); tmp = f.name
    got = {n for _, n in check(tmp)}
    os.unlink(tmp)
    ok = got == {"C", "D"}
    print(f"{'✅' if ok else '⛔'} self-test：抓到 {sorted(got)}，預期 ['C', 'D']")
    sys.exit(0 if ok else 1)

bad = 0
for path in sys.argv[1:]:
    ps = check(path)
    if ps:
        bad = 1
        print(f"⛔ {path}")
        for line, name in sorted(ps):
            print(f"   line {line}: 用到 `{name}` 但它還沒定義")
    else:
        print(f"✅ {path}")
sys.exit(bad)
