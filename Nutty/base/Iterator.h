#pragma once

namespace nutty
{
    template <
        typename T, 
        typename Container
    >
    class Iterator
    {
        typedef T* pointer;
        typedef Container* container_pointer;
        typedef const T& const_type_reference;
        typedef T& type_reference;
        typedef size_t size_type;

    private:
        container_pointer m_container;
        pointer m_ptr;

    public:
        Iterator(void)
            : m_ptr(NULL), m_container(NULL)
        {
        }

        Iterator(pointer ptr, container_pointer container) 
            : m_ptr(ptr), m_container(container)
        {
        }

        Iterator& operator+=(size_type i)
        {
            m_ptr += i;
            return *this;
        }

        Iterator& operator-=(size_type i)
        {
            m_ptr -= i;
            return *this;
        }

        Iterator operator++(void)
        {
            Iterator it(m_ptr, m_container);
            m_ptr++;
            return it;
        }

        Iterator& operator++(int)
        {
            m_ptr++;
            return *this;
        }

        const_type_reference operator=(const_type_reference t)
        {
            m_container->Insert(*this, t);
            return t;
        }

        bool operator==(const Iterator& it)
        {
            return m_ptr == it.m_ptr;
        }

        bool operator!=(const Iterator& it)
        {
            return m_ptr != it.m_ptr;
        }

        pointer operator()(void)
        {
            return m_ptr;
        }

        pointer operator()(void) const
        {
            return m_ptr;
        }

        T operator*(void)
        {
            return (*m_container)[*this];
        }

        template <
            typename T, 
            typename Container
        >
        friend Iterator<T, Container> operator+(Iterator<T, Container>& i0, Iterator<T, Container>& i1);

        template <
            typename T, 
            typename Container
        >
        friend Iterator<T, Container> operator-(Iterator<T, Container>& i0, Iterator<T, Container>& i1);

        template <
            typename T,
            typename Container
        >
        friend Iterator<T, Container> operator+(Iterator<T, Container>& i0, size_type i);

        template <
            typename T,
            typename Container
        >
        friend Iterator<T, Container> operator-(Iterator<T, Container>& i0, size_type i);
    };

    template <
        typename T, 
        typename Container
    >
    Iterator<T, Container> operator+(Iterator<T, Container>& i0, Iterator<T, Container>& i1)
    {
        assert(i0.m_container == i1.m_container);
        T* p = i0.m_ptr + i1.m_ptr;
        Iterator<T, Container> it(p, i0.m_container);
        return it;
    }

    template <
        typename T, 
        typename Container
    >
    Iterator<T, Container> operator-(Iterator<T, Container>& i0, Iterator<T, Container>& i1)
    {
        assert(i0.m_container == i1.m_container);
        T* p = i0.m_ptr - i1.m_ptr;
        Iterator<T, Container> it(p, i0.m_container);
        return it;
    }

    template <
        typename T,
        typename Container
    >
    Iterator<T, Container> operator+(Iterator<T, Container>& i0, typename Iterator<T, Container>::size_type i)
    {
        T* p = i0.m_ptr + i;
        Iterator<T, Container> it(p, i0.m_container);
        return it;
    }

    template <
        typename T,
        typename Container
    >
    bool operator<(Iterator<T, Container>& i0, Iterator<T, Container>& i1)
    {
        return i0.m_ptr < i1.m_ptr;
    }

    template <
        typename T, 
        typename Container
    >
    Iterator<T, Container> operator-(Iterator<T, Container>& i0, typename Iterator<T, Container>::size_type i)
    {
        T* p = i0.m_ptr - i;
        Iterator<T, Container> it(p, i0.m_container);
        return it;
    }

    template <
        typename T,
        typename C
    >
    size_t Distance(const Iterator<T, C>& begin, const Iterator<T, C>& end)
    {
        return (end() - begin());
    }

    template <
        typename T,
        typename C
    >
    size_t Distance(Iterator<T, C>& begin, Iterator<T, C>& end)
    {
        return (end() - begin());
    }
}