import random
import string
import re

# Удаление комментариев
def remove_comments(js_code):
    # Регулярное выражение для удаления многострочных комментариев
    multiline_comment_pattern = r'/\*[\s\S]*?\*/'
    # Регулярное выражение для удаления однострочных комментариев
    singleline_comment_pattern = r'//.*'
    
    # Удаление многострочных комментариев
    js_code = re.sub(multiline_comment_pattern, '', js_code, flags=re.DOTALL)
    # Удаление однострочных комментариев
    js_code = re.sub(singleline_comment_pattern, '', js_code, flags=re.MULTILINE)
    
    return js_code.strip();

# Минимизация пробелов
def minimize_spaces(js_code):
    lines = js_code.split('\n')
    return ''.join(line.strip() for line in lines)


# Генерация нового имени для переменной
def generate_random_name(length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

# Обфускация переменных
def obfuscate_variables(js_code):
    var_map = {}
    
    for var_name in re.findall(r'\bvar\s+(\w+)', js_code):
        obfuscated_name = generate_random_name()
        var_map[var_name] = obfuscated_name

    for original_name, obfuscated_name in var_map.items():
        js_code = re.sub(r'\b' + original_name + r'\b', obfuscated_name, js_code)

    return js_code

# Генерация "мертвого" кода
def add_dead_code(js_code):
    dead_code = f"function {generate_random_name()}(){{console.log(’{generate_random_name()}');}}\n"
    return dead_code + js_code

def obfuscate_js(js_code):
    js_code = remove_comments(js_code)
    js_code = obfuscate_variables(js_code)
    js_code = minimize_spaces(js_code)
    #js_code = add_dead_code(js_code)
    return js_code

original_js = """ здесь код, необходимый обфусцировать """
obfuscated_js = obfuscate_js(original_js)
print(obfuscated_js)
