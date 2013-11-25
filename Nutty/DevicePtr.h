#pragma once
namespace nutty
{
    template <
        typename T
    >
    class DevicePtr
    {
        typedef T* pointer;
        typedef size_t size_type;

    private:
        pointer m_ptr;

    public:
        DevicePtr(const DevicePtr& p) : m_ptr(p.m_ptr) { }

        DevicePtr(pointer ptr) : m_ptr(ptr) { }

        pointer* GetRawPointerPtr(void) { return &m_ptr; }

        pointer operator()(void) const { return m_ptr; }

        template <
            typename T
        >
        friend DevicePtr<T> operator+(DevicePtr<T>& i0, DevicePtr<T>& i1);

        template <
            typename T
        >
        friend DevicePtr<T> operator-(DevicePtr<T>& i0, DevicePtr<T>& i1);

        template <
            typename T
        >
        friend DevicePtr<T> operator+(DevicePtr<T>& i0, size_type i);

        template <
            typename T
        >
        friend DevicePtr<T> operator-(DevicePtr<T>& i0, size_type i);
    };

    template <
        typename T
    >
    DevicePtr<T> operator+(DevicePtr<T>& i0, DevicePtr<T>& i1)
    {
        T* p = i0.m_ptr + i1.m_ptr;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    DevicePtr<T> operator-(DevicePtr<T>& i0, DevicePtr<T>& i1)
    {
        T* p = i0.m_ptr - i1.m_ptr;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    DevicePtr<T> operator+(DevicePtr<T>& i0, typename DevicePtr<T>::size_type i)
    {
        T* p = i0.m_ptr + i;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    DevicePtr<T> operator-(DevicePtr<T>& i0, typename DevicePtr<T>::size_type i)
    {
        T* p = i0.m_ptr - i;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    size_t Distance(const DevicePtr<T>& begin, const DevicePtr<T>& end)
    {
        return (end() - begin());
    }

    template <
        typename T
    >
    size_t Distance(DevicePtr<T>& begin, DevicePtr<T>& end)
    {
        return (end() - begin());
    }

    template <
        typename T,
        typename Iterator
    >
    size_t Distance(Iterator& begin, DevicePtr<T>& end)
    {
        return (end() - begin());
    }

    template <
        typename T,
        typename Iterator
    >
    size_t Distance(DevicePtr<T>& begin, Iterator& end)
    {
        return (end() - begin());
    }


    template <
        typename T
    >
    DevicePtr<T> DevicePtr_Cast(T* ptr)
    {
        return DevicePtr<T>(ptr);
    }
}