
import os
from non_personalised import non_personalised_rc
from CF_personalised import cf_personalised
from bcolour import bcolours

# Function to clear the console
def clearConsole():
    os.system('cls' if os.name == 'nt' else 'clear')
    
# if not found this will be raised
class MovieIdNotFoundError(Exception):
    pass

# welcome page
def welcomePage():
    clearConsole()
    print(bcolours.HEADER,bcolours.BOLD + "##             Welcome to the MovieMatch              ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "##                                                     ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "## This program will help you find a movie to watch    ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "##                                                     ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "## based on your" + bcolours.ENDC, bcolours.UNDERLINE,bcolours.FAIL + "past behavior and preferences" + bcolours.ENDC,bcolours.OKBLUE  + "       ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "##                                                     ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "## and also with " + bcolours.ENDC, bcolours.UNDERLINE,bcolours.FAIL + "current hot trends!" + bcolours.ENDC,bcolours.OKBLUE+"                ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "##                                                     ##" + bcolours.ENDC)
    print(bcolours.OKGREEN + "## Ready to explore? Let's begin your movie journey!   ##" + bcolours.ENDC)
    print(bcolours.OKBLUE + "##                                                     ##" + bcolours.ENDC)
    user_input = input(bcolours.WARNING + "Please enter your user id to continue: " + bcolours.ENDC)
    return user_input

# check if user id is valid or active
def checkUser(user_input):
    if int(user_input) in range(1,6041):
        return True
    else:
        return False

# enjoy movie page
def enjoyMovie():
    clearConsole()
    print(bcolours.HEADER + "##  Perfect choice!                                  " + bcolours.ENDC)
    print(bcolours.FAIL + "##  Light is down, stage is set!                 " + bcolours.ENDC)
    print(bcolours.OKGREEN + bcolours.BOLD + "##  Enjoy your movie!                                " + bcolours.ENDC)

# recommend movie page
def recommendMovie(non_personalised_movies, personalised_movies):
    count = 0
    while True:
        clearConsole()
        not_over = True
        if count+5 >= len(non_personalised_movies):
            not_over = False
            print(bcolours.HEADER + "##  Trending Now                                     ##" + bcolours.ENDC)
            for each in non_personalised_movies.iloc[count:].itertuples():
                print( bcolours.OKCYAN + "MovieId: " + bcolours.ENDC, str(each.MovieID) + bcolours.OKCYAN + " :: Movie name: " + bcolours.ENDC + str(each.Title)  +  bcolours.OKCYAN + " :: Movie genres: " + bcolours.ENDC + str(each.Genres))
            print("##                                                   ##")
            print( bcolours.HEADER + "##  Top Picks for you                                ##" + bcolours.ENDC)
            print("##                                                   ##")
            for each in personalised_movies.iloc[count:].itertuples():
                print(bcolours.OKGREEN + "MovieId: " + bcolours.ENDC + str(each.MovieID) + bcolours.OKGREEN + " :: Movie name: " + bcolours.ENDC + str(each.Title)  + bcolours.OKGREEN + " :: Movie genres: " + bcolours.ENDC + str(each.Genres))
            print(bcolours.FAIL + "##  End of the page                                  ##" + bcolours.ENDC)
            count = 0
            
        if not_over:
            print("##                                                   ##")
            print(bcolours.HEADER + "##  Trending Now                                     ##" + bcolours.ENDC)
            for each in non_personalised_movies.iloc[count:count+5].itertuples():
                print(bcolours.OKCYAN + "MovieId: " + bcolours.ENDC + str(each.MovieID) + bcolours.OKCYAN + " :: Movie name: " + bcolours.ENDC + str(each.Title)  + bcolours.OKCYAN + " :: Movie genres: " + bcolours.ENDC + str(each.Genres))
            print("##                                                   ##")
            print(bcolours.HEADER + "##  Top Picks for you                                ##" + bcolours.ENDC)
            print("##                                                   ##")
            for each in personalised_movies.iloc[count:count+5].itertuples():
                print(bcolours.OKGREEN + "MovieId: " + bcolours.ENDC + str(each.MovieID) + bcolours.OKGREEN + " :: Movie name: " + bcolours.ENDC + str(each.Title)  + bcolours.OKGREEN + " :: Movie genres: " + bcolours.ENDC + str(each.Genres))
                
        print("##                                                   ##")
        print(bcolours.OKGREEN + "##  Would you like to watch any of these movies?     ##" + bcolours.ENDC)
        if not_over:
            user_input = input(bcolours.WARNING + "Enter movie ID to enjoy the movie, if want to see more options, enter c to continue explore: " + bcolours.ENDC)
        else:
            user_input = input(bcolours.FAIL + "Enter movie ID to enjoy the movie, this is end of the page, enter c to continue explore: " + bcolours.ENDC)
               
        if user_input == "c":
            if not_over:
                count += 10
        else:
            try:
                user_input = int(user_input)
                if int(user_input) in non_personalised_movies["MovieID"].values or int(user_input) in personalised_movies["MovieID"].values:
                    clearConsole()
                    enjoyMovie()
                    break
                elif int(user_input) not in personalised_movies["MovieID"].values:
                    raise MovieIdNotFoundError("Invalid movie ID. Please enter a valid movie ID or c to continue explore.")
            except ValueError:
                input(bcolours.FAIL + "I am sorry, I didn't get that. Please enter a valid movie ID or c to continue explore." + bcolours.ENDC)
                continue
            except MovieIdNotFoundError:
                input(bcolours.FAIL + "Invalid movie ID. Please enter a valid movie ID or c to continue explore." + bcolours.ENDC)

# wrong user id page
def wrongUser():
    print( bcolours.FAIL +"##  Sorry, we can't find your user id.                ##" + bcolours.ENDC)
    print( bcolours.FAIL +"##  Please try again.                          ##" + bcolours.ENDC)

# main function
def main():
    user_input = welcomePage()
    if checkUser(user_input):
        clearConsole()
        print( bcolours.HEADER,bcolours.BOLD + "##  Welcome back, USER:" + bcolours.ENDC,bcolours.OKBLUE,bcolours.ITALIC + str(user_input) + bcolours.ENDC,bcolours.HEADER,bcolours.BOLD + "                      ##" + bcolours.ENDC)
        print( bcolours.OKGREEN + "##  Let's find a movie for you!                      ##" + bcolours.ENDC)
        print( bcolours.WARNING + "##  Processing...                                    ##" + bcolours.ENDC)
        user_id = int(user_input)
        non_personalised_movies = non_personalised_rc(user_id,100)
        personalised_movies = cf_personalised(user_id,100)
        recommendMovie(non_personalised_movies=non_personalised_movies, personalised_movies=personalised_movies)
    else:
        clearConsole()
        wrongUser()

main()