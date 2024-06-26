# MarioMapVisualizer

This function turns strings, where each char corresponds to a block, into a mario map pngs.
It was written in quiet a rush so it is not of the best quality.
If you would like to add new blocks, modify the dictionary in the beginning of the code, modify the spritesheet and finally should it be necessary add new exceptions where the sprites are drawn.
  
Symbol References taken from 
https://github.com/amidos2006/Mario-AI-Framework/blob/master/README.md
## Requirements
- python
- python-pygame (to create the canvas using the sprites)

## Examples

### Input 1:
```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------g-----------------------------------------------------------------------------------------------------------------------
----------------------!---------------------------------------------------------SSSSSSSS---SSS!--------------@-----------SSS----S!!S--------------------------------------------------------##------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###------------
-------------------------------------------------------------------------------g----------------------------------------------------------------------------------------------------------####------------
----------------------------------------------TT----------T-----1------------------------------------------------------------------------------------------------------------------------#####------------
----------------!---S@S!S-------------T-------()---------()------------------S@S--------------C-----SU----!--!--!-----S----------SS------#--#----------##--#------------SS!S------------######------------
--------------------------------------()------[]---------[]-----------------------------------------------------------------------------##--##--------###--##--------------------------#######------------
----------------------------()--------[]------[]---------[]----------------------------------------------------------------------------###--###------####--###-----()--------------()-########--------F---
---M-----------------g------[]--------[]-g----[]-----g-g-[]------------------------------------g-g--------k-----------------gg-g-g----####--####----#####--####----[]---------gg---[]#########--------#---
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXX---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXX---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
### Output 1:
<p align="center" >
<img src="https://github.com/MisterNimbus/StringToMarioMapPNG/blob/main/ExampleWorld1Result.png" width="100%" />
</p>


### Input 2:
```
------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------oo--------------------------------------------------------------------------------------------------------------------
---------------------------------------g-g------------------------------------------------------------------------------------------------------------
----------------------ooor---------%%%%%%%-------------oooo----------R------------------oo--oo--------------------------------------------------------
---------------------%%%%%----------|||||--------------%%%%----------------g----oo---------------------------R-----oo--------------------##-----------
----------------------|||-----------|||||----oo---SSSS--||-------------%%%%%%----------------------------r-------------------------------##-----------
----------------------|||-----------|||||---------------||--------------||||-----------------------%%%%%%%%----------------------------####-----------
----------------------|||-----%%%%%-|||||---------------||--------------||||-----SSS----------------||||||---------------------SSS-----####-----------
-------------------%%%%%%%%----|||--|||||---------------||-------%%%----||||-------------SSS--------||||||-----%%%%--%%%%------------######-----------
--------------------||||||-----|||--|||||-------------U-||--------|-----||||------------------------||||||------||----||-------------######-----------
--------------------||||||--o--|||--|||||---------------||--------|-----||||-----------------%%%%---||||||------||----||-------------######--------F--
-------------%%%%---||||||-%%%-|||--|||||---------------||--------|-----||||------------------||----||||||--ooo-||----||---------r---######--------#--
XXXXXXXXXXX---||----||||||--|--|||--|||||----%%%%-----%%%%%-%%%%%-|-----||||------------------||----||||||--%%%-||----||----XXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXX---||----||||||--|--|||--|||||-----||-------|||---|||--|-----||||------------------||----||||||---|--||----||----XXXXXXXXXXXXXXXXXXXXXXXXXX
```
### Output 2:
<p align="center" >
<img src="https://github.com/MisterNimbus/StringToMarioMapPNG/blob/main/ExampleWorld2Result.png" width="100%" />
</p>

<h3 id="copyrights">Copyrights</h3>

------
This function is not endorsed by Nintendo and is only intended for research purposes. Mario is a Nintendo character which the authors don't own any rights to. Nintendo is also the sole owner of all the graphical assets used. Any use of this function is expected to be on a non-commercial basis. This function was created as a tool to ease the visualization world creation research.
