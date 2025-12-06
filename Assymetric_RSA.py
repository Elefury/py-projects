def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def modinv(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def encrypt(message, e, n):
    return [pow(ord(char), e, n) for char in message]

def decrypt(ciphertext, d, n):
    return ''.join([chr(pow(char, d, n)) for char in ciphertext])

# Генерация ключей
p = 61
q = 53
n = p * q
phi = (p - 1) * (q - 1)
e = 17
d = modinv(e, phi)

with open('input.txt', 'r') as file:
    plaintext = file.read()

ciphertext = encrypt(plaintext, e, n)

# Шифрование текста в файл ciphertext.txt
with open('ciphertext.txt', 'w') as file:
    file.write(' '.join(map(str, ciphertext)))

with open('ciphertext.txt', 'r') as file:
    ciphertext = list(map(int, file.read().split()))

decrypted_text = decrypt(ciphertext, d, n)

# Расшифрованный текст в decrypted_text.txt
with open('decrypted_text.txt', 'w') as file:
    file.write(decrypted_text)
