def maximize_number(A, B):

    list_A = list(A)
    list_B = sorted(B, reverse=True)      
    pointer_B = 0

    for i in range(len(list_A)):
        if pointer_B < len(list_B) and list_B[pointer_B] > list_A[i]:
            list_A[i] = list_B[pointer_B]
            pointer_B += 1
    return ''.join(list_A)

while True:
    print("\nВведите данные (или 'exit' для выхода):")
    A = input("Число A: ").strip()
    
    if A.lower() == 'exit':
        print("Выход из программы.")
        break
        
    B = input("Число B: ").strip()
    
    if B.lower() == 'exit':
        print("Выход из программы.")
        break
    
    if not A.isdigit() or not B.isdigit():
        print("Ошибка: введите только цифры!")
        continue
    
    result = maximize_number(A, B)
    
    print(f"\nРезультат:")
    print(f"Исходное A: {A}")
    print(f"Цифры из B: {B}")
    print(f"Макс. A: {result}")
    print("-" * 30)
