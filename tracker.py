expenses = []

#helps create a simple expense tracker that allows users to add expenses, view them, and calculate the total amount spent. The program uses a list to store expenses, where each expense is represented as a dictionary containing the name, category, and amount of the expense. The user can interact with the program through a simple menu-driven interface.
def add_expense():
    name = input("Enter the name of the expense: ")
    category = input("Enter the category of the expense: ")
    amount = float(input("amount: "))
    expense = {
        "name": name,
        "category": category,
        "amount": amount
    }
    expenses.append(expense)
    print("Expense added successfully!")

def view_expenses():
    if not expenses:
        print("No expenses recorded.")
        return
    for idx, expense in enumerate(expenses, start=1):
        print(f"{idx}. {expense['name']} - {expense['category']} - ${expense['amount']:.2f}")

def show_total():
    total = sum(expense['amount'] for expense in expenses)
    print(f"Total expenses: ${total:.2f}")

def main():
    while True:
        print("\nExpense Tracker")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Show Total")
        print("4. Exit")
        choice = input("Choose an option: ")
        
        if choice == '1':
            add_expense()
        elif choice == '2':
            view_expenses()
        elif choice == '3':
            show_total()
        elif choice == '4':
            print("Exiting the expense tracker. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
    
if __name__ == "__main__":
    main()