Author(s): <mfernandezcarmona@lincoln.ac.uk>


#darknet-service, a python-service for NAOqi

This service is based upon the template *python-service* from the project robot-jumpstarter. See it doc for more information on creating services.

It requires darknet to be installed on the system, with python support. See repo.

Contrary to the usual, this application WONT BE INSTALLED ON THE ROBOT, as it lacks from GPU to run it. 

We will use the development approach, that allows to run it in another computer.

(**`python app/scripts/darknetsrv.py --qi-url  [your robot's IP]:[robot port]`**)
Remember to  
`source ~/spqrel/workspace/spqrel_launch/worktree/spqrel_tools/setup-dev.bash`
`source ~/spqrel/workspace/spqrel_launch/worktree/spqrel_tools/setup.bash`

This project also contains unit tests: run `python testrun2.py --ip [robot ip] --port [robot port]` in the project root (this is experimental).

See also
========

 * [The Official Python SDK documentation](http://doc.aldebaran.com/2-4/dev/python/).
 * The [Studio Toolkit libraries](https://github.com/pepperhacking/studiotoolkit/), used in these templates (stk, and robotutils.js)
 * [Notes on "Services"](/doc/services.md), an attempt to clarify some terminological confusion
