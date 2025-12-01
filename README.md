# CS-NEA-AERO
Aerofoil tester using a fluid simulation and shape drawer

Currently have a (seemingly) functional LBM simulation, along with an aerofoil drawer. 



TODO:
finish menu stuff:
- sim settings
- propulsion device menu (link up buttons)
add propulsion device sims:
- add a way to use preset simulations (internal boundary shape and conditions) for prop devices
add unit conversions:
- calculator to convert from inputted SI units to sim units
- calculator to convert from ouputted sim units force to newtons

add settings calculator:
- calculator to calculate density from given temperature and altitude (with assumtions listed in writeup)
- calculator to calculate likely air temperature given a density (assuming an altitude of 12200m and ^^ assumptions)

### Images:

main menu

<img width="600" height="550" alt="Screenshot 2025-11-20 201417" src="https://github.com/user-attachments/assets/39182978-a203-4acf-91f3-1a13b10edd45" />

drawing a shape

<img width="600" height="550" alt="Screenshot 2025-11-20 201342" src="https://github.com/user-attachments/assets/25c5ca81-14f3-4c93-872f-4d6990b6a957" />

early sim curl plot
<img width="1241" height="550" alt="Screenshot 2025-11-20 201548" src="https://github.com/user-attachments/assets/153faa8d-9830-4126-a02d-7fa2d650b03d" />

early sim velocity plot
<img width="1242" height="551" alt="Screenshot 2025-11-20 201558" src="https://github.com/user-attachments/assets/016293d0-3a8a-4c16-9bcf-9019eb9e4418" />

mid sim curl plot
<img width="1244" height="551" alt="Screenshot 2025-11-20 201735" src="https://github.com/user-attachments/assets/b9dbcf60-4305-44f8-8e4a-f08b2f438fe3" />

later sim curl plot
<img width="1236" height="553" alt="Screenshot 2025-11-20 203714" src="https://github.com/user-attachments/assets/0e3c28de-b649-4fca-9059-581ba8e6b704" />

late sim curl plot (convergence occured)
<img width="1241" height="557" alt="Screenshot 2025-11-20 204319" src="https://github.com/user-attachments/assets/0c6ca167-e1b8-4639-86bb-8abae58f39b8" />

~20k iterations curl plot
<img width="1243" height="553" alt="Screenshot 2025-11-20 204725" src="https://github.com/user-attachments/assets/882f769d-24da-4807-b7ac-31bcdfb4a67b" />

~20k iterations velocity plot
<img width="1244" height="557" alt="Screenshot 2025-11-20 204734" src="https://github.com/user-attachments/assets/5ea791e4-364c-4e58-a071-45edb9e80c72" />

~25k iterations curl plot
<img width="1244" height="555" alt="image" src="https://github.com/user-attachments/assets/10449640-6bde-4354-8046-c6162a876010" />

At this point, the simulation slows down considerably. Not entirely sure why. The graph has half its points deleted every time it reaches 5000 points, which allowed me to get to 25k, but beyond that it began buffering every frmae
