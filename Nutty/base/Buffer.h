#pragma once
#include "Iterator.h"
#include "../Alloc.h"

namespace nutty
{
    namespace base
    {
        template<
            typename T,
            typename Content,
            typename Allocator = DefaultAllocator<T>
        >
        class Base_Buffer
        {
        private:
            Base_Buffer(const Base_Buffer& b) {}

        public:
            Base_Buffer(Base_Buffer&& c) : m_ptr(NULL), m_size(0)
            {
                m_ptr = c.m_ptr;
                m_size = c.m_size;

                c.m_ptr = NULL;
                c.m_size = 0;
            }

            Base_Buffer& operator=(Base_Buffer&& c)
            {
                if(this != &c)
                {
                    Clear();

                    m_ptr = c.m_ptr;
                    m_size = c.m_size;
                    
                    c.m_ptr = NULL;
                    c.m_size = 0;
                }

                return *this;
            }

            typedef T* pointer;
            typedef const T& const_type_reference;
            typedef const pointer const_pointer;
            typedef T& type_reference;
            typedef Iterator<T, Base_Buffer<T, Content, Allocator>> iterator;
            typedef const Iterator<T, Base_Buffer<T, Content, Allocator>> const_iterator;
            typedef size_t size_type;
            typedef const size_type& const_size_typ_reference;

            Base_Buffer(void)
            {
                m_size = 0;
                m_ptr = 0;
            }

            Base_Buffer(size_type n)
            {
                m_size = n;
                m_ptr = 0;
                Resize(m_size);
            }

            Base_Buffer(size_type n, const_type_reference t)
            {
                m_size = n;
                m_ptr = 0;
                Resize(m_size);
                Fill(Begin(), End(), t);
            }

            iterator Begin(void)
            {
                //assert(m_size > 0);
                return iterator(m_ptr, this);
            }

            iterator End(void)
            {
                return iterator(m_ptr + m_size, this);
            }

            const_iterator Begin(void) const
            {
                return const_iterator(m_ptr, this);
            }

            const_iterator End(void) const
            {
                return const_iterator(m_ptr + m_size, this);
            }

            bool Empty(void)
            {
                return m_size == 0;
            }

            void Resize(size_type n)
            {
                T* old = m_ptr;
                size_type cpySize = min(n, m_size);
                m_ptr = m_alloc.Allocate(n);
                if(old)
                {
                    Copy(Begin(), const_iterator(old, this), cpySize);
                    m_alloc.Deallocate(old);
                }
                m_size = n;
            }

            void Clear(void)
            {
                if(m_ptr)
                {
                    m_alloc.Deallocate(m_ptr);
                }
                m_size = 0;
                m_ptr = NULL;
            }

            const_size_typ_reference Size(void) const
            {
                return m_size;
            }

            void Insert(iterator pos, const_type_reference v)
            {
                m_content.Insert(pos, v);
            }

            void Insert(iterator start, iterator end, const_type_reference v)
            {
                m_content.Insert(start, end, v);
            }

            void Insert(size_type pos, const_type_reference v)
            {
                Insert(Begin() + pos, v);
            }

            T operator[](iterator it)
            {
                return m_content[it];
            }

            T operator[](size_type index)
            {
                assert(index < m_size);
                iterator it = Begin();
                it += index;
                return operator[](it);
            }

            pointer* GetRawPointer(void)
            {
                return &m_ptr;
            }

            /*pointer GetRawPointer(void)
            {
                return m_ptr;
            }*/

            virtual ~Base_Buffer(void)
            {
                Clear();
            }

        protected:
            size_t m_size;
            Allocator m_alloc;
            pointer m_ptr;
            Content m_content;
        };
    }
}