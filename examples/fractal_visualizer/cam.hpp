#include "keyboard.hpp"

const int SQUARES = MEM_SIZE;
void onIdle() {

  glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(-SQUARES/2, -SQUARES/2, -100.0));
  
  //  float xrotrad, yrotrad;
  //  yrotrad = (yrot / 180 * 3.141592654f);
  //  xrotrad = (xrot / 180 * 3.141592654f);

  glm::mat4 view = glm::lookAt(glm::vec3(Camera.Position.x, Camera.Position.y, Camera.Position.z),
      glm::vec3(Camera.Position.x + Camera.ViewDir.x,Camera.Position.y + Camera.ViewDir.y, Camera.Position.z + Camera.ViewDir.z),
      glm::vec3(Camera.UpVector.x, Camera.UpVector.y, Camera.UpVector.z));
  glm::mat4 projection = glm::perspective(40.0f, 1.0f*screen_width/screen_height, 0.1f, 1000.0f);

  glm::mat4 mvp = projection * view * model;// * anim;
  glUseProgram(program);
  glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
  glutPostRedisplay();
}
