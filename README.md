# CS-NEA-AERO
Aerofoil tester using a fluid simulation and shape drawer

Currently have a (seemingly) functional LBM simulation, along with an aerofoil drawer. 



TODO:
testing and bugfixing
If I decide to continue the project after NEA, will create a new project with the same starting codebase
This would be for my own enjoyment, and would more or less do whatever I feel like doing

# Images:

## main menu

<img width="600" height="550" alt="Screenshot 2025-11-20 201417" src="https://github.com/user-attachments/assets/39182978-a203-4acf-91f3-1a13b10edd45" />

## drawing a shape

<img width="600" height="550" alt="Screenshot 2025-11-20 201342" src="https://github.com/user-attachments/assets/25c5ca81-14f3-4c93-872f-4d6990b6a957" />

## simulating using the drawn aerofoil

#### early sim curl plot
<img width="1241" height="550" alt="Screenshot 2025-11-20 201548" src="https://github.com/user-attachments/assets/153faa8d-9830-4126-a02d-7fa2d650b03d" />

#### early sim velocity plot
<img width="1242" height="551" alt="Screenshot 2025-11-20 201558" src="https://github.com/user-attachments/assets/016293d0-3a8a-4c16-9bcf-9019eb9e4418" />

#### mid sim curl plot
<img width="1244" height="551" alt="Screenshot 2025-11-20 201735" src="https://github.com/user-attachments/assets/b9dbcf60-4305-44f8-8e4a-f08b2f438fe3" />

#### later sim curl plot
<img width="1236" height="553" alt="Screenshot 2025-11-20 203714" src="https://github.com/user-attachments/assets/0e3c28de-b649-4fca-9059-581ba8e6b704" />

#### late sim curl plot (convergence occured)
<img width="1241" height="557" alt="Screenshot 2025-11-20 204319" src="https://github.com/user-attachments/assets/0c6ca167-e1b8-4639-86bb-8abae58f39b8" />

#### ~20k iterations curl plot
<img width="1243" height="553" alt="Screenshot 2025-11-20 204725" src="https://github.com/user-attachments/assets/882f769d-24da-4807-b7ac-31bcdfb4a67b" />

#### ~20k iterations velocity plot
<img width="1244" height="557" alt="Screenshot 2025-11-20 204734" src="https://github.com/user-attachments/assets/5ea791e4-364c-4e58-a071-45edb9e80c72" />

#### ~25k iterations curl plot
<img width="1244" height="555" alt="image" src="https://github.com/user-attachments/assets/10449640-6bde-4354-8046-c6162a876010" />

## other menus

#### Simulations
<img width="752" height="790" alt="Screenshot 2026-01-03 205659" src="https://github.com/user-attachments/assets/74bfe532-6a18-4654-b44b-2281c52e6b8e" />

#### Aerofoils list
<img width="749" height="790" alt="Screenshot 2026-01-03 205713" src="https://github.com/user-attachments/assets/53c64faf-3794-43d7-8835-a53b6e58b439" />

#### Propulsion Simulations
<img width="753" height="788" alt="Screenshot 2026-01-03 205734" src="https://github.com/user-attachments/assets/319f3154-35ac-4f0a-b040-ca718806d7df" />

#### simulation settings
<img width="755" height="793" alt="Screenshot 2026-01-03 205751" src="https://github.com/user-attachments/assets/321f8bae-9f04-4e58-ae84-d1e5bc17d938" />

The values can be edited, with changing altitude and temperature calculating a density, and entering a density calculating an altitude and temperature
<img width="752" height="790" alt="Screenshot 2026-01-03 205823" src="https://github.com/user-attachments/assets/fe42d9eb-74f4-4d6a-9859-5d91f3f9c3b8" />

There is input validation
<img width="751" height="794" alt="Screenshot 2026-01-03 205906" src="https://github.com/user-attachments/assets/e8399d88-e3bd-475c-9ca1-db343a3c17a4" />

## Simulating Propeller

#### early simulation velocity plot
<img width="754" height="415" alt="Screenshot 2026-01-05 214944" src="https://github.com/user-attachments/assets/c79ce889-25b0-47e3-8bfa-8e20cb6d0976" />

#### early simulation curl plot
<img width="750" height="421" alt="Screenshot 2026-01-05 214929" src="https://github.com/user-attachments/assets/1e492130-d6ed-4a57-b652-7e9c1c5ab0b6" />

#### late simulation velocity plot
<img width="1404" height="680" alt="image" src="https://github.com/user-attachments/assets/72c63d1a-ec36-44c5-b92e-691adfcc9d2a" />

#### late simulation curl plot
<img width="1403" height="682" alt="image" src="https://github.com/user-attachments/assets/a44bdc97-6adf-40f9-9d98-a875c92e7a38" />
There seems to be an issue with the propeller simulation. At around 7000 iterations, the simulation 'blows up'. Not too sure why. Even at only 3000 iterations (slightly before the images shown), the simulation looks radically different from how it perhaps should.

## Simulating Jet Engine

#### early simulation velocity plot
<img width="754" height="417" alt="Screenshot 2026-01-05 222358" src="https://github.com/user-attachments/assets/7a7ec4ad-d3a8-4a85-9d41-58ddba16bb11" />

#### early simulation curl plot
<img width="755" height="416" alt="Screenshot 2026-01-05 222334" src="https://github.com/user-attachments/assets/2ceadec1-29fa-4059-b1ee-ea96fe690c6b" />

#### late simulation velocity plot
<img width="1400" height="690" alt="Screenshot 2026-01-05 225528" src="https://github.com/user-attachments/assets/5f5b2f80-bf8b-4b5c-8401-f7d4c8898165" />

#### late simulation curl plot
<img width="1401" height="687" alt="image" src="https://github.com/user-attachments/assets/b2738ac6-efac-4e94-a5b8-b77d7266b946" />

