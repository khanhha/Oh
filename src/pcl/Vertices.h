#ifndef PCL_MESSAGE_VERTICES_H
#define PCL_MESSAGE_VERTICES_H
#include <string>
#include <vector>
#include <ostream>
#include <memory.h>
#include <pcl/pcl_macros.h>

namespace pcl
{
  /** \brief Describes a set of vertices in a polygon mesh, by basically
    * storing an array of indices.
    */
  struct Vertices
  {
    Vertices () : vertices ()
    {}

    std::vector<uint32_t> vertices;

  public:
    typedef std::shared_ptr<Vertices> Ptr;
    typedef std::shared_ptr<Vertices const> ConstPtr;
  }; // struct Vertices


  typedef std::shared_ptr<Vertices> VerticesPtr;
  typedef std::shared_ptr<Vertices const> VerticesConstPtr;

  inline std::ostream& operator<<(std::ostream& s, const  ::pcl::Vertices & v)
  {
    s << "vertices[]" << std::endl;
    for (size_t i = 0; i < v.vertices.size (); ++i)
    {
      s << "  vertices[" << i << "]: ";
      s << "  " << v.vertices[i] << std::endl;
    }
    return (s);
  }
} // namespace pcl

#endif // PCL_MESSAGE_VERTICES_H

