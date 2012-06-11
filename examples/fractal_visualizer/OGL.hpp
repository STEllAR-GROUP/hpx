#include "cam.hpp"
#include <vector>
int init_resources()
{
    
    GLfloat cube_vertices[SQUARES * 9 * 2 * SQUARES];
    {
        std::vector<GLfloat> verts;
    int diff = (MEM_SIZE/SQUARES);
    
    float scaling = 100;
        for (int i = 0; i < SQUARES; i++)
            for (int j = 0; j < SQUARES; j++)
            {
                bool toggle = false;
                if ((i + 1)*diff > MEM_SIZE-1 || (j + 1)*diff > MEM_SIZE-1)
                    toggle = true;
                else if ((i)*diff - 3 < 0 || (j)*diff - 3 < 0)
                    toggle = true;
                //0,0 //0 + i * SQUARES + j
                verts.push_back(i*2);   
                verts.push_back(j);     
                if (!toggle)
                {
                    float set = 0;
                    for (int x = 0; x < diff; x++)
                        for (int y = 0; y < diff; y++)
                        {
                            set += SetImage.GetPixel(i*diff-x,j*diff-y).Blue/(diff*scaling);
                        }
                    verts.push_back(set);
                }
                else
                    verts.push_back(0);
                //1,0 //1 + i * SQUARES + j
                verts.push_back((i+1)*2);   
                verts.push_back(j);
                if (!toggle)
                {
                    float set = 0;
                    for (int x = 0; x < diff; x++)
                        for (int y = 0; y < diff; y++)
                        {
                            set += SetImage.GetPixel((i+1)*diff-x,(j)*diff-y).Blue/(diff*scaling);
                        }
                    verts.push_back(set);
                }
                else
                    verts.push_back(0);
                //0,1 //2 + i * SQUARES + j
                verts.push_back(i*2);
                verts.push_back(j+1);
                if (!toggle)
                {
                    float set = 0;
                    for (int x = 0; x < diff; x++)
                        for (int y = 0; y < diff; y++)
                        {
                            set += SetImage.GetPixel((i)*diff-x,(j+1)*diff-y).Blue/(diff*scaling);
                        }
                    verts.push_back(set);
                }
                else
                    verts.push_back(0);

                //1,1 //3 + i * SQUARES + j
                verts.push_back((i+1)*2);
                verts.push_back(j+1);
                if (!toggle)
                {
                    float set = 0;
                    for (int x = 0; x < diff; x++)
                        for (int y = 0; y < diff; y++)
                        {
                            set += SetImage.GetPixel((i+1)*diff-x,(j+1)*diff-y).Blue/(diff*scaling);
                        }
                    verts.push_back(set);
                }
                else
                    verts.push_back(0);
                //1,0 //4 + i * SQUARES + j
                verts.push_back((i+1)*2);
                verts.push_back(j);
                if (!toggle)
                {
                    float set = 0;
                    for (int x = 0; x < diff; x++)
                        for (int y = 0; y < diff; y++)
                        {
                            set += SetImage.GetPixel((i+1)*diff-x,(j)*diff-y).Blue/(diff*scaling);
                        }
                    verts.push_back(set);
                }
                else
                    verts.push_back(0);
                //0,1 //5 + i * SQUARES + j
                verts.push_back(i*2);
                verts.push_back(j+1);
                if (!toggle)
                {
                    float set = 0;
                    for (int x = 0; x < diff; x++)
                        for (int y = 0; y < diff; y++)
                        {
                            set += SetImage.GetPixel((i)*diff-x,(j+1)*diff-y).Blue/(diff*scaling);
                        }
                    verts.push_back(set);
                }
                else
                    verts.push_back(0);
            }
        for(int i = 0; i < verts.size(); i++)
            cube_vertices[i] = verts[i];
        verts.empty();
        SetImage.SetSize(1,1);
  glGenBuffers(1, &vbo_cube_vertices);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
  glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
   }
  GLfloat cube_texcoords[SQUARES*SQUARES*2*6];
  {
      float div = SQUARES;
      std::vector<GLfloat> texs;
      for (int i = 0; i < SQUARES; i++)
            for (int j = 0; j < SQUARES; j++)
              {
                  float x = (float)i+0.5;
                  float y = (float)j+0.5;
                    //0,0
                  texs.push_back(x/div);
                  texs.push_back(y/div);
                    //1,0
                  texs.push_back((x+1)/div);
                  texs.push_back(y/div);
                    //0,1
                  texs.push_back(x/div);
                  texs.push_back((y+1)/div);

                    //1,1
                  texs.push_back((x+1)/div);
                  texs.push_back((y+1)/div);
                    //1,0
                  texs.push_back((x+1)/div);
                  texs.push_back((y)/div);
                    //0,1
                  texs.push_back((x)/div);
                  texs.push_back((y+1)/div);
              }
        for(int i = 0; i < texs.size(); i++)
            cube_texcoords[i] = texs[i];
        texs.empty();
  }
  glGenBuffers(1, &vbo_cube_texcoords);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_texcoords);
  glBufferData(GL_ARRAY_BUFFER, sizeof(cube_texcoords), cube_texcoords, GL_STATIC_DRAW);

	IMAGE decalImage;
	decalImage.Load("out.bmp");
	decalImage.ExpandPalette();
  glGenTextures(1, &texture_id);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  
	glTexImage2D(	GL_TEXTURE_2D, 0, GL_RGBA8, decalImage.width, decalImage.height,
					0, decalImage.format, GL_UNSIGNED_BYTE, decalImage.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  GLint link_ok = GL_FALSE;

  GLuint vs, fs;
  if ((vs = create_shader("mesh.v.glsl", GL_VERTEX_SHADER))   == 0) return 0;
  if ((fs = create_shader("mesh.f.glsl", GL_FRAGMENT_SHADER)) == 0) return 0;

  program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
  if (!link_ok) {
    fprintf(stderr, "glLinkProgram:");
    print_log(program);
    return 0;
  }

  const char* attribute_name;
  attribute_name = "coord3d";
  attribute_coord3d = glGetAttribLocation(program, attribute_name);
  if (attribute_coord3d == -1) {
    fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
    return 0;
  }
  attribute_name = "texcoord";
  attribute_texcoord = glGetAttribLocation(program, attribute_name);
  if (attribute_texcoord == -1) {
    fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
    return 0;
  }
  const char* uniform_name;
  uniform_name = "mvp";
  uniform_mvp = glGetUniformLocation(program, uniform_name);
  if (uniform_mvp == -1) {
    fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
    return 0;
  }
  uniform_name = "mytexture";
  uniform_mytexture = glGetUniformLocation(program, uniform_name);
  if (uniform_mytexture == -1) {
    fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
    return 0;
  }

  return 1;
}


void onDisplay()
{
  glClearColor(0.5, 0.9, 0.1, 1.0);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  glUseProgram(program);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glUniform1i(uniform_mytexture, /*GL_TEXTURE*/0);

  glEnableVertexAttribArray(attribute_coord3d);
  // Describe our vertices array to OpenGL (it can't guess its format automatically)
  glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
  glVertexAttribPointer(
    attribute_coord3d, // attribute
    3,                 // number of elements per vertex, here (x,y,z)
    GL_FLOAT,          // the type of each element
    GL_FALSE,          // take our values as-is
    0,                 // no extra data between each position
    0                  // offset of first element
  );

  glEnableVertexAttribArray(attribute_texcoord);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_texcoords);
  glVertexAttribPointer(
    attribute_texcoord, // attribute
    2,                  // number of elements per vertex, here (x,y)
    GL_FLOAT,           // the type of each element
    GL_FALSE,           // take our values as-is
    0,                  // no extra data between each position
    0                   // offset of first element
  );
  glDrawArrays(GL_TRIANGLES, 0, SQUARES*6*SQUARES);

  glDisableVertexAttribArray(attribute_coord3d);
  glDisableVertexAttribArray(attribute_texcoord);
  glutSwapBuffers();
}

void onReshape(int width, int height) {
  screen_width = width;
  screen_height = height;
  glViewport(0, 0, screen_width, screen_height);
}

void free_resources()
{
  glDeleteProgram(program);
  glDeleteBuffers(1, &vbo_cube_vertices);
  glDeleteBuffers(1, &vbo_cube_texcoords);
  glDeleteBuffers(1, &ibo_cube_elements);
  glDeleteTextures(1, &texture_id);
}
