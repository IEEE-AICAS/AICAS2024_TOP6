function t(){throw new Error("Cycle detected")}const n=Symbol.for("preact-signals");function e(){if(f>1){f--;return}let t,n=!1;while(void 0!==o){let _=o;o=void 0;l++;while(void 0!==_){const i=_.o;_.o=void 0;_.f&=-3;if(!(8&_.f)&&p(_))try{_.c()}catch(e){if(!n){t=e;n=!0}}_=i}}l=0;f--;if(n)throw t}function _(t){if(f>0)return t();f++;try{return t()}finally{e()}}let i,o,r=0;function u(t){if(r>0)return t();const n=i;i=void 0;r++;try{return t()}finally{r--;i=n}}let f=0,l=0,s=0;function c(t){if(void 0===i)return;let n=t.n;if(void 0===n||n.t!==i){n={i:0,S:t,p:i.s,n:void 0,t:i,e:void 0,x:void 0,r:n};if(void 0!==i.s)i.s.n=n;i.s=n;t.n=n;if(32&i.f)t.S(n);return n}else if(-1===n.i){n.i=0;if(void 0!==n.n){n.n.p=n.p;if(void 0!==n.p)n.p.n=n.n;n.p=i.s;n.n=void 0;i.s.n=n;i.s=n}return n}}function h(t){this.v=t;this.i=0;this.n=void 0;this.t=void 0}h.prototype.brand=n;h.prototype.h=function(){return!0};h.prototype.S=function(t){if(this.t!==t&&void 0===t.e){t.x=this.t;if(void 0!==this.t)this.t.e=t;this.t=t}};h.prototype.U=function(t){if(void 0!==this.t){const n=t.e,e=t.x;if(void 0!==n){n.x=e;t.e=void 0}if(void 0!==e){e.e=n;t.x=void 0}if(t===this.t)this.t=e}};h.prototype.subscribe=function(t){const n=this;return w((function(){const e=n.value,_=32&this.f;this.f&=-33;try{t(e)}finally{this.f|=_}}))};h.prototype.valueOf=function(){return this.value};h.prototype.toString=function(){return this.value+""};h.prototype.toJSON=function(){return this.value};h.prototype.peek=function(){return this.v};Object.defineProperty(h.prototype,"value",{get(){const t=c(this);if(void 0!==t)t.i=this.i;return this.v},set(n){if(i instanceof y)!function(){throw new Error("Computed cannot have side-effects")}();if(n!==this.v){if(l>100)t();this.v=n;this.i++;s++;f++;try{for(let t=this.t;void 0!==t;t=t.x)t.t.N()}finally{e()}}}});function a(t){return new h(t)}function p(t){for(let n=t.s;void 0!==n;n=n.n)if(n.S.i!==n.i||!n.S.h()||n.S.i!==n.i)return!0;return!1}function d(t){for(let n=t.s;void 0!==n;n=n.n){const e=n.S.n;if(void 0!==e)n.r=e;n.S.n=n;n.i=-1;if(void 0===n.n){t.s=n;break}}}function v(t){let n,e=t.s;while(void 0!==e){const t=e.p;if(-1===e.i){e.S.U(e);if(void 0!==t)t.n=e.n;if(void 0!==e.n)e.n.p=t}else n=e;e.S.n=e.r;if(void 0!==e.r)e.r=void 0;e=t}t.s=n}function y(t){h.call(this,void 0);this.x=t;this.s=void 0;this.g=s-1;this.f=4}(y.prototype=new h).h=function(){this.f&=-3;if(1&this.f)return!1;if(32==(36&this.f))return!0;this.f&=-5;if(this.g===s)return!0;this.g=s;this.f|=1;if(this.i>0&&!p(this)){this.f&=-2;return!0}const t=i;try{d(this);i=this;const t=this.x();if(16&this.f||this.v!==t||0===this.i){this.v=t;this.f&=-17;this.i++}}catch(t){this.v=t;this.f|=16;this.i++}i=t;v(this);this.f&=-2;return!0};y.prototype.S=function(t){if(void 0===this.t){this.f|=36;for(let t=this.s;void 0!==t;t=t.n)t.S.S(t)}h.prototype.S.call(this,t)};y.prototype.U=function(t){if(void 0!==this.t){h.prototype.U.call(this,t);if(void 0===this.t){this.f&=-33;for(let t=this.s;void 0!==t;t=t.n)t.S.U(t)}}};y.prototype.N=function(){if(!(2&this.f)){this.f|=6;for(let t=this.t;void 0!==t;t=t.x)t.t.N()}};y.prototype.peek=function(){if(!this.h())t();if(16&this.f)throw this.v;return this.v};Object.defineProperty(y.prototype,"value",{get(){if(1&this.f)t();const n=c(this);this.h();if(void 0!==n)n.i=this.i;if(16&this.f)throw this.v;return this.v}});function m(t){return new y(t)}function g(t){const n=t.u;t.u=void 0;if("function"==typeof n){f++;const _=i;i=void 0;try{n()}catch(n){t.f&=-2;t.f|=8;b(t);throw n}finally{i=_;e()}}}function b(t){for(let n=t.s;void 0!==n;n=n.n)n.S.U(n);t.x=void 0;t.s=void 0;g(t)}function k(t){if(i!==this)throw new Error("Out-of-order effect");v(this);i=t;this.f&=-2;if(8&this.f)b(this);e()}function S(t){this.x=t;this.u=void 0;this.s=void 0;this.o=void 0;this.f=32}S.prototype.c=function(){const t=this.S();try{if(8&this.f)return;if(void 0===this.x)return;const n=this.x();if("function"==typeof n)this.u=n}finally{t()}};S.prototype.S=function(){if(1&this.f)t();this.f|=1;this.f&=-9;g(this);d(this);f++;const n=i;i=this;return k.bind(this,n)};S.prototype.N=function(){if(!(2&this.f)){this.f|=2;this.o=o;o=this}};S.prototype.d=function(){this.f|=8;if(!(1&this.f))b(this)};function w(t){const n=new S(t);try{n.c()}catch(t){n.d();throw t}return n.d.bind(n)}var x,C,E,U,H,P,N,$,D,T={},V=[],A=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i,F=Array.isArray;function M(t,n){for(var e in n)t[e]=n[e];return t}function W(t){var n=t.parentNode;n&&n.removeChild(t)}function L(t,n,e){var _,i,o,r={};for(o in n)"key"==o?_=n[o]:"ref"==o?i=n[o]:r[o]=n[o];if(arguments.length>2&&(r.children=arguments.length>3?x.call(arguments,2):e),"function"==typeof t&&null!=t.defaultProps)for(o in t.defaultProps)void 0===r[o]&&(r[o]=t.defaultProps[o]);return O(t,r,_,i,null)}function O(t,n,e,_,i){var o={type:t,props:n,key:e,ref:_,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,constructor:void 0,__v:null==i?++E:i,__i:-1,__u:0};return null==i&&null!=C.vnode&&C.vnode(o),o}function R(){return{current:null}}function j(t){return t.children}function I(t,n){this.props=t,this.context=n}function q(t,n){if(null==n)return t.__?q(t.__,t.__i+1):null;for(var e;n<t.__k.length;n++)if(null!=(e=t.__k[n])&&null!=e.__e)return e.__e;return"function"==typeof t.type?q(t):null}function B(t,n,e){var _,i=t.__v,o=i.__e,r=t.__P;if(r)return(_=M({},i)).__v=i.__v+1,C.vnode&&C.vnode(_),it(r,_,i,t.__n,void 0!==r.ownerSVGElement,32&i.__u?[o]:null,n,null==o?q(i):o,!!(32&i.__u),e),_.__v=i.__v,_.__.__k[_.__i]=_,_.__d=void 0,_.__e!=o&&G(_),_}function G(t){var n,e;if(null!=(t=t.__)&&null!=t.__c){for(t.__e=t.__c.base=null,n=0;n<t.__k.length;n++)if(null!=(e=t.__k[n])&&null!=e.__e){t.__e=t.__c.base=e.__e;break}return G(t)}}function z(t){(!t.__d&&(t.__d=!0)&&H.push(t)&&!J.__r++||P!==C.debounceRendering)&&((P=C.debounceRendering)||N)(J)}function J(){var t,n,e,_=[],i=[];for(H.sort($);t=H.shift();)t.__d&&(e=H.length,n=B(t,_,i)||n,0===e||H.length>e?(ot(_,n,i),i.length=_.length=0,n=void 0,H.sort($)):n&&C.__c&&C.__c(n,V));n&&ot(_,n,i),J.__r=0}function K(t,n,e,_,i,o,r,u,f,l,s){var c,h,a,p,d,v=_&&_.__k||V,y=n.length;for(e.__d=f,Q(e,n,v),f=e.__d,c=0;c<y;c++)null!=(a=e.__k[c])&&"boolean"!=typeof a&&"function"!=typeof a&&(h=-1===a.__i?T:v[a.__i]||T,a.__i=c,it(t,a,h,i,o,r,u,f,l,s),p=a.__e,a.ref&&h.ref!=a.ref&&(h.ref&&ut(h.ref,null,a),s.push(a.ref,a.__c||p,a)),null==d&&null!=p&&(d=p),65536&a.__u||h.__k===a.__k?f=X(a,f,t):"function"==typeof a.type&&void 0!==a.__d?f=a.__d:p&&(f=p.nextSibling),a.__d=void 0,a.__u&=-196609);e.__d=f,e.__e=d}function Q(t,n,e){var _,i,o,r,u,f=n.length,l=e.length,s=l,c=0;for(t.__k=[],_=0;_<f;_++)r=_+c,null!=(i=t.__k[_]=null==(i=n[_])||"boolean"==typeof i||"function"==typeof i?null:"string"==typeof i||"number"==typeof i||"bigint"==typeof i||i.constructor==String?O(null,i,null,null,null):F(i)?O(j,{children:i},null,null,null):void 0===i.constructor&&i.__b>0?O(i.type,i.props,i.key,i.ref?i.ref:null,i.__v):i)?(i.__=t,i.__b=t.__b+1,u=Z(i,e,r,s),i.__i=u,o=null,-1!==u&&(s--,(o=e[u])&&(o.__u|=131072)),null==o||null===o.__v?(-1==u&&c--,"function"!=typeof i.type&&(i.__u|=65536)):u!==r&&(u===r+1?c++:u>r?s>f-r?c+=u-r:c--:u<r?u==r-1&&(c=u-r):c=0,u!==_+c&&(i.__u|=65536))):(o=e[r])&&null==o.key&&o.__e&&0==(131072&o.__u)&&(o.__e==t.__d&&(t.__d=q(o)),ft(o,o,!1),e[r]=null,s--);if(s)for(_=0;_<l;_++)null!=(o=e[_])&&0==(131072&o.__u)&&(o.__e==t.__d&&(t.__d=q(o)),ft(o,o))}function X(t,n,e){var _,i;if("function"==typeof t.type){for(_=t.__k,i=0;_&&i<_.length;i++)_[i]&&(_[i].__=t,n=X(_[i],n,e));return n}t.__e!=n&&(e.insertBefore(t.__e,n||null),n=t.__e);do{n=n&&n.nextSibling}while(null!=n&&8===n.nodeType);return n}function Y(t,n){return n=n||[],null==t||"boolean"==typeof t||(F(t)?t.some((function(t){Y(t,n)})):n.push(t)),n}function Z(t,n,e,_){var i=t.key,o=t.type,r=e-1,u=e+1,f=n[e];if(null===f||f&&i==f.key&&o===f.type&&0==(131072&f.__u))return e;if(_>(null!=f&&0==(131072&f.__u)?1:0))for(;r>=0||u<n.length;){if(r>=0){if((f=n[r])&&0==(131072&f.__u)&&i==f.key&&o===f.type)return r;r--}if(u<n.length){if((f=n[u])&&0==(131072&f.__u)&&i==f.key&&o===f.type)return u;u++}}return-1}function tt(t,n,e){"-"===n[0]?t.setProperty(n,null==e?"":e):t[n]=null==e?"":"number"!=typeof e||A.test(n)?e:e+"px"}function nt(t,n,e,_,i){var o;t:if("style"===n)if("string"==typeof e)t.style.cssText=e;else{if("string"==typeof _&&(t.style.cssText=_=""),_)for(n in _)e&&n in e||tt(t.style,n,"");if(e)for(n in e)_&&e[n]===_[n]||tt(t.style,n,e[n])}else if("o"===n[0]&&"n"===n[1])o=n!==(n=n.replace(/(PointerCapture)$|Capture$/i,"$1")),n=n.toLowerCase()in t?n.toLowerCase().slice(2):n.slice(2),t.l||(t.l={}),t.l[n+o]=e,e?_?e.u=_.u:(e.u=Date.now(),t.addEventListener(n,o?_t:et,o)):t.removeEventListener(n,o?_t:et,o);else{if(i)n=n.replace(/xlink(H|:h)/,"h").replace(/sName$/,"s");else if("width"!==n&&"height"!==n&&"href"!==n&&"list"!==n&&"form"!==n&&"tabIndex"!==n&&"download"!==n&&"rowSpan"!==n&&"colSpan"!==n&&"role"!==n&&n in t)try{t[n]=null==e?"":e;break t}catch(t){}"function"==typeof e||(null==e||!1===e&&"-"!==n[4]?t.removeAttribute(n):t.setAttribute(n,e))}}function et(t){if(this.l){var n=this.l[t.type+!1];if(t.t){if(t.t<=n.u)return}else t.t=Date.now();return n(C.event?C.event(t):t)}}function _t(t){if(this.l)return this.l[t.type+!0](C.event?C.event(t):t)}function it(t,n,e,_,i,o,r,u,f,l){var s,c,h,a,p,d,v,y,m,g,b,k,S,w,x,E=n.type;if(void 0!==n.constructor)return null;128&e.__u&&(f=!!(32&e.__u),o=[u=n.__e=e.__e]),(s=C.__b)&&s(n);t:if("function"==typeof E)try{if(y=n.props,m=(s=E.contextType)&&_[s.__c],g=s?m?m.props.value:s.__:_,e.__c?v=(c=n.__c=e.__c).__=c.__E:("prototype"in E&&E.prototype.render?n.__c=c=new E(y,g):(n.__c=c=new I(y,g),c.constructor=E,c.render=lt),m&&m.sub(c),c.props=y,c.state||(c.state={}),c.context=g,c.__n=_,h=c.__d=!0,c.__h=[],c._sb=[]),null==c.__s&&(c.__s=c.state),null!=E.getDerivedStateFromProps&&(c.__s==c.state&&(c.__s=M({},c.__s)),M(c.__s,E.getDerivedStateFromProps(y,c.__s))),a=c.props,p=c.state,c.__v=n,h)null==E.getDerivedStateFromProps&&null!=c.componentWillMount&&c.componentWillMount(),null!=c.componentDidMount&&c.__h.push(c.componentDidMount);else{if(null==E.getDerivedStateFromProps&&y!==a&&null!=c.componentWillReceiveProps&&c.componentWillReceiveProps(y,g),!c.__e&&(null!=c.shouldComponentUpdate&&!1===c.shouldComponentUpdate(y,c.__s,g)||n.__v===e.__v)){for(n.__v!==e.__v&&(c.props=y,c.state=c.__s,c.__d=!1),n.__e=e.__e,n.__k=e.__k,n.__k.forEach((function(t){t&&(t.__=n)})),b=0;b<c._sb.length;b++)c.__h.push(c._sb[b]);c._sb=[],c.__h.length&&r.push(c);break t}null!=c.componentWillUpdate&&c.componentWillUpdate(y,c.__s,g),null!=c.componentDidUpdate&&c.__h.push((function(){c.componentDidUpdate(a,p,d)}))}if(c.context=g,c.props=y,c.__P=t,c.__e=!1,k=C.__r,S=0,"prototype"in E&&E.prototype.render){for(c.state=c.__s,c.__d=!1,k&&k(n),s=c.render(c.props,c.state,c.context),w=0;w<c._sb.length;w++)c.__h.push(c._sb[w]);c._sb=[]}else do{c.__d=!1,k&&k(n),s=c.render(c.props,c.state,c.context),c.state=c.__s}while(c.__d&&++S<25);c.state=c.__s,null!=c.getChildContext&&(_=M(M({},_),c.getChildContext())),h||null==c.getSnapshotBeforeUpdate||(d=c.getSnapshotBeforeUpdate(a,p)),K(t,F(x=null!=s&&s.type===j&&null==s.key?s.props.children:s)?x:[x],n,e,_,i,o,r,u,f,l),c.base=n.__e,n.__u&=-161,c.__h.length&&r.push(c),v&&(c.__E=c.__=null)}catch(t){n.__v=null,f||null!=o?(n.__e=u,n.__u|=f?160:32,o[o.indexOf(u)]=null):(n.__e=e.__e,n.__k=e.__k),C.__e(t,n,e)}else null==o&&n.__v===e.__v?(n.__k=e.__k,n.__e=e.__e):n.__e=rt(e.__e,n,e,_,i,o,r,f,l);(s=C.diffed)&&s(n)}function ot(t,n,e){for(var _=0;_<e.length;_++)ut(e[_],e[++_],e[++_]);C.__c&&C.__c(n,t),t.some((function(n){try{t=n.__h,n.__h=[],t.some((function(t){t.call(n)}))}catch(t){C.__e(t,n.__v)}}))}function rt(t,n,e,_,i,o,r,u,f){var l,s,c,h,a,p,d,v=e.props,y=n.props,m=n.type;if("svg"===m&&(i=!0),null!=o)for(l=0;l<o.length;l++)if((a=o[l])&&"setAttribute"in a==!!m&&(m?a.localName===m:3===a.nodeType)){t=a,o[l]=null;break}if(null==t){if(null===m)return document.createTextNode(y);t=i?document.createElementNS("http://www.w3.org/2000/svg",m):document.createElement(m,y.is&&y),o=null,u=!1}if(null===m)v===y||u&&t.data===y||(t.data=y);else{if(o=o&&x.call(t.childNodes),v=e.props||T,!u&&null!=o)for(v={},l=0;l<t.attributes.length;l++)v[(a=t.attributes[l]).name]=a.value;for(l in v)a=v[l],"children"==l||("dangerouslySetInnerHTML"==l?c=a:"key"===l||l in y||nt(t,l,null,a,i));for(l in y)a=y[l],"children"==l?h=a:"dangerouslySetInnerHTML"==l?s=a:"value"==l?p=a:"checked"==l?d=a:"key"===l||u&&"function"!=typeof a||v[l]===a||nt(t,l,a,v[l],i);if(s)u||c&&(s.__html===c.__html||s.__html===t.innerHTML)||(t.innerHTML=s.__html),n.__k=[];else if(c&&(t.innerHTML=""),K(t,F(h)?h:[h],n,e,_,i&&"foreignObject"!==m,o,r,o?o[0]:e.__k&&q(e,0),u,f),null!=o)for(l=o.length;l--;)null!=o[l]&&W(o[l]);u||(l="value",void 0!==p&&(p!==t[l]||"progress"===m&&!p||"option"===m&&p!==v[l])&&nt(t,l,p,v[l],!1),l="checked",void 0!==d&&d!==t[l]&&nt(t,l,d,v[l],!1))}return t}function ut(t,n,e){try{"function"==typeof t?t(n):t.current=n}catch(t){C.__e(t,e)}}function ft(t,n,e){var _,i;if(C.unmount&&C.unmount(t),(_=t.ref)&&(_.current&&_.current!==t.__e||ut(_,null,n)),null!=(_=t.__c)){if(_.componentWillUnmount)try{_.componentWillUnmount()}catch(t){C.__e(t,n)}_.base=_.__P=null,t.__c=void 0}if(_=t.__k)for(i=0;i<_.length;i++)_[i]&&ft(_[i],n,e||"function"!=typeof t.type);e||null==t.__e||W(t.__e),t.__=t.__e=t.__d=void 0}function lt(t,n,e){return this.constructor(t,e)}function st(t,n,e){var _,i,o,r;C.__&&C.__(t,n),i=(_="function"==typeof e)?null:e&&e.__k||n.__k,o=[],r=[],it(n,t=(!_&&e||n).__k=L(j,null,[t]),i||T,T,void 0!==n.ownerSVGElement,!_&&e?[e]:i?null:n.firstChild?x.call(n.childNodes):null,o,!_&&e?e:i?i.__e:n.firstChild,_,r),t.__d=void 0,ot(o,t,r)}function ct(t,n){st(t,n,ct)}function ht(t,n,e){var _,i,o,r,u=M({},t.props);for(o in t.type&&t.type.defaultProps&&(r=t.type.defaultProps),n)"key"==o?_=n[o]:"ref"==o?i=n[o]:u[o]=void 0===n[o]&&void 0!==r?r[o]:n[o];return arguments.length>2&&(u.children=arguments.length>3?x.call(arguments,2):e),O(t.type,u,_||t.key,i||t.ref,null)}function at(t,n){var e={__c:n="__cC"+D++,__:t,Consumer:function(t,n){return t.children(n)},Provider:function(t){var e,_;return this.getChildContext||(e=[],(_={})[n]=this,this.getChildContext=function(){return _},this.shouldComponentUpdate=function(t){this.props.value!==t.value&&e.some((function(t){t.__e=!0,z(t)}))},this.sub=function(t){e.push(t);var n=t.componentWillUnmount;t.componentWillUnmount=function(){e.splice(e.indexOf(t),1),n&&n.call(t)}}),t.children}};return e.Provider.__=e.Consumer.contextType=e}x=V.slice,C={__e:function(t,n,e,_){for(var i,o,r;n=n.__;)if((i=n.__c)&&!i.__)try{if((o=i.constructor)&&null!=o.getDerivedStateFromError&&(i.setState(o.getDerivedStateFromError(t)),r=i.__d),null!=i.componentDidCatch&&(i.componentDidCatch(t,_||{}),r=i.__d),r)return i.__E=i}catch(n){t=n}throw t}},E=0,U=function(t){return null!=t&&null==t.constructor},I.prototype.setState=function(t,n){var e;e=null!=this.__s&&this.__s!==this.state?this.__s:this.__s=M({},this.state),"function"==typeof t&&(t=t(M({},e),this.props)),t&&M(e,t),null!=t&&this.__v&&(n&&this._sb.push(n),z(this))},I.prototype.forceUpdate=function(t){this.__v&&(this.__e=!0,t&&this.__h.push(t),z(this))},I.prototype.render=j,H=[],N="function"==typeof Promise?Promise.prototype.then.bind(Promise.resolve()):setTimeout,$=function(t,n){return t.__v.__b-n.__v.__b},J.__r=0,D=0;var pt,dt,vt,yt,mt=0,gt=[],bt=[],kt=C,St=kt.__b,wt=kt.__r,xt=kt.diffed,Ct=kt.__c,Et=kt.unmount,Ut=kt.__;function Ht(t,n){kt.__h&&kt.__h(dt,t,mt||n),mt=0;var e=dt.__H||(dt.__H={__:[],__h:[]});return t>=e.__.length&&e.__.push({__V:bt}),e.__[t]}function Pt(t){return mt=1,Nt(zt,t)}function Nt(t,n,e){var _=Ht(pt++,2);if(_.t=t,!_.__c&&(_.__=[e?e(n):zt(void 0,n),function(t){var n=_.__N?_.__N[0]:_.__[0],e=_.t(n,t);n!==e&&(_.__N=[e,_.__[1]],_.__c.setState({}))}],_.__c=dt,!dt.u)){var i=function(t,n,e){if(!_.__c.__H)return!0;var i=_.__c.__H.__.filter((function(t){return!!t.__c}));if(i.every((function(t){return!t.__N})))return!o||o.call(this,t,n,e);var r=!1;return i.forEach((function(t){if(t.__N){var n=t.__[0];t.__=t.__N,t.__N=void 0,n!==t.__[0]&&(r=!0)}})),!(!r&&_.__c.props===t)&&(!o||o.call(this,t,n,e))};dt.u=!0;var o=dt.shouldComponentUpdate,r=dt.componentWillUpdate;dt.componentWillUpdate=function(t,n,e){if(this.__e){var _=o;o=void 0,i(t,n,e),o=_}r&&r.call(this,t,n,e)},dt.shouldComponentUpdate=i}return _.__N||_.__}function $t(t,n){var e=Ht(pt++,3);!kt.__s&&Gt(e.__H,n)&&(e.__=t,e.i=n,dt.__H.__h.push(e))}function Dt(t,n){var e=Ht(pt++,4);!kt.__s&&Gt(e.__H,n)&&(e.__=t,e.i=n,dt.__h.push(e))}function Tt(t){return mt=5,At((function(){return{current:t}}),[])}function Vt(t,n,e){mt=6,Dt((function(){return"function"==typeof t?(t(n()),function(){return t(null)}):t?(t.current=n(),function(){return t.current=null}):void 0}),null==e?e:e.concat(t))}function At(t,n){var e=Ht(pt++,7);return Gt(e.__H,n)?(e.__V=t(),e.i=n,e.__h=t,e.__V):e.__}function Ft(t,n){return mt=8,At((function(){return t}),n)}function Mt(t){var n=dt.context[t.__c],e=Ht(pt++,9);return e.c=t,n?(null==e.__&&(e.__=!0,n.sub(dt)),n.props.value):t.__}function Wt(t,n){kt.useDebugValue&&kt.useDebugValue(n?n(t):t)}function Lt(t){var n=Ht(pt++,10),e=Pt();return n.__=t,dt.componentDidCatch||(dt.componentDidCatch=function(t,_){n.__&&n.__(t,_),e[1](t)}),[e[0],function(){e[1](void 0)}]}function Ot(){var t=Ht(pt++,11);if(!t.__){for(var n=dt.__v;null!==n&&!n.__m&&null!==n.__;)n=n.__;var e=n.__m||(n.__m=[0,0]);t.__="P"+e[0]+"-"+e[1]++}return t.__}function Rt(){for(var t;t=gt.shift();)if(t.__P&&t.__H)try{t.__H.__h.forEach(qt),t.__H.__h.forEach(Bt),t.__H.__h=[]}catch(n){t.__H.__h=[],kt.__e(n,t.__v)}}kt.__b=function(t){dt=null,St&&St(t)},kt.__=function(t,n){t&&n.__k&&n.__k.__m&&(t.__m=n.__k.__m),Ut&&Ut(t,n)},kt.__r=function(t){wt&&wt(t),pt=0;var n=(dt=t.__c).__H;n&&(vt===dt?(n.__h=[],dt.__h=[],n.__.forEach((function(t){t.__N&&(t.__=t.__N),t.__V=bt,t.__N=t.i=void 0}))):(n.__h.forEach(qt),n.__h.forEach(Bt),n.__h=[],pt=0)),vt=dt},kt.diffed=function(t){xt&&xt(t);var n=t.__c;n&&n.__H&&(n.__H.__h.length&&(1!==gt.push(n)&&yt===kt.requestAnimationFrame||((yt=kt.requestAnimationFrame)||It)(Rt)),n.__H.__.forEach((function(t){t.i&&(t.__H=t.i),t.__V!==bt&&(t.__=t.__V),t.i=void 0,t.__V=bt}))),vt=dt=null},kt.__c=function(t,n){n.some((function(t){try{t.__h.forEach(qt),t.__h=t.__h.filter((function(t){return!t.__||Bt(t)}))}catch(u){n.some((function(t){t.__h&&(t.__h=[])})),n=[],kt.__e(u,t.__v)}})),Ct&&Ct(t,n)},kt.unmount=function(t){Et&&Et(t);var n,e=t.__c;e&&e.__H&&(e.__H.__.forEach((function(t){try{qt(t)}catch(t){n=t}})),e.__H=void 0,n&&kt.__e(n,e.__v))};var jt="function"==typeof requestAnimationFrame;function It(t){var n,e=function(){clearTimeout(_),jt&&cancelAnimationFrame(n),setTimeout(t)},_=setTimeout(e,100);jt&&(n=requestAnimationFrame(e))}function qt(t){var n=dt,e=t.__c;"function"==typeof e&&(t.__c=void 0,e()),dt=n}function Bt(t){var n=dt;t.__c=t.__(),dt=n}function Gt(t,n){return!t||t.length!==n.length||n.some((function(n,e){return n!==t[e]}))}function zt(t,n){return"function"==typeof n?n(t):n}function Jt(t,n){C[t]=n.bind(null,C[t]||(()=>{}))}let Kt,Qt;function Xt(t){if(Qt)Qt();Qt=t&&t.S()}function Yt({data:t}){const n=tn(t);n.value=t;const e=At(()=>{let t=this.__v;while(t=t.__)if(t.__c){t.__c.__$f|=4;break}this.__$u.c=()=>{var t;if(!U(e.peek())&&3===(null==(t=this.base)?void 0:t.nodeType))this.base.data=e.peek();else{this.__$f|=1;this.setState({})}};return m(()=>{let t=n.value.value;return 0===t?0:!0===t?"":t||""})},[]);return e.value}Yt.displayName="_st";Object.defineProperties(h.prototype,{constructor:{configurable:!0,value:void 0},type:{configurable:!0,value:Yt},props:{configurable:!0,get(){return{data:this}}},__b:{configurable:!0,value:1}});Jt("__b",(t,n)=>{if("string"==typeof n.type){let t,e=n.props;for(let _ in e){if("children"===_)continue;let i=e[_];if(i instanceof h){if(!t)n.__np=t={};t[_]=i;e[_]=i.peek()}}}t(n)});Jt("__r",(t,n)=>{Xt();let e,_=n.__c;if(_){_.__$f&=-2;e=_.__$u;if(void 0===e)_.__$u=e=function(t){let n;w((function(){n=this}));n.c=()=>{_.__$f|=1;_.setState({})};return n}()}Kt=_;Xt(e);t(n)});Jt("__e",(t,n,e,_)=>{Xt();Kt=void 0;t(n,e,_)});Jt("diffed",(t,n)=>{Xt();Kt=void 0;let e;if("string"==typeof n.type&&(e=n.__e)){let t=n.__np,_=n.props;if(t){let n=e.U;if(n)for(let e in n){let _=n[e];if(void 0!==_&&!(e in t)){_.d();n[e]=void 0}}else{n={};e.U=n}for(let i in t){let o=n[i],r=t[i];if(void 0===o){o=Zt(e,i,r,_);n[i]=o}else o.o(r,_)}}}t(n)});function Zt(t,n,e,_){const i=n in t&&void 0===t.ownerSVGElement,o=a(e);return{o:(t,n)=>{o.value=t;_=n},d:w(()=>{const e=o.value.value;if(_[n]!==e){_[n]=e;if(i)t[n]=e;else if(e)t.setAttribute(n,e);else t.removeAttribute(n)}})}}Jt("unmount",(t,n)=>{if("string"==typeof n.type){let t=n.__e;if(t){const n=t.U;if(n){t.U=void 0;for(let t in n){let e=n[t];if(e)e.d()}}}}else{let t=n.__c;if(t){const n=t.__$u;if(n){t.__$u=void 0;n.d()}}}t(n)});Jt("__h",(t,n,e,_)=>{if(_<3||9===_)n.__$f|=2;t(n,e,_)});I.prototype.shouldComponentUpdate=function(t,n){const e=this.__$u;if(!(e&&void 0!==e.s||4&this.__$f))return!0;if(3&this.__$f)return!0;for(let _ in n)return!0;for(let _ in t)if("__source"!==_&&t[_]!==this.props[_])return!0;for(let _ in this.props)if(!(_ in t))return!0;return!1};function tn(t){return At(()=>a(t),[])}function nn(t){const n=Tt(t);n.current=t;Kt.__$f|=4;return At(()=>m(()=>n.current()),[])}function en(t){const n=Tt(t);n.current=t;$t(()=>w(()=>n.current()),[])}var _n=function(t,n,e,_){var i;n[0]=0;for(var o=1;o<n.length;o++){var r=n[o++],u=n[o]?(n[0]|=r?1:2,e[n[o++]]):n[++o];3===r?_[0]=u:4===r?_[1]=Object.assign(_[1]||{},u):5===r?(_[1]=_[1]||{})[n[++o]]=u:6===r?_[1][n[++o]]+=u+"":r?(i=t.apply(u,_n(t,u,e,["",null])),_.push(i),u[0]?n[0]|=2:(n[o-2]=0,n[o]=i)):_.push(u)}return _},on=new Map;function rn(t){var n=on.get(this);return n||(n=new Map,on.set(this,n)),(n=_n(this,n.get(t)||(n.set(t,n=function(t){for(var n,e,_=1,i="",o="",r=[0],u=function(t){1===_&&(t||(i=i.replace(/^\s*\n\s*|\s*\n\s*$/g,"")))?r.push(0,t,i):3===_&&(t||i)?(r.push(3,t,i),_=2):2===_&&"..."===i&&t?r.push(4,t,0):2===_&&i&&!t?r.push(5,0,!0,i):_>=5&&((i||!t&&5===_)&&(r.push(_,0,i,e),_=6),t&&(r.push(_,t,0,e),_=6)),i=""},f=0;f<t.length;f++){f&&(1===_&&u(),u(f));for(var l=0;l<t[f].length;l++)n=t[f][l],1===_?"<"===n?(u(),r=[r],_=3):i+=n:4===_?"--"===i&&">"===n?(_=1,i=""):i=n+i[0]:o?n===o?o="":i+=n:'"'===n||"'"===n?o=n:">"===n?(u(),_=1):_&&("="===n?(_=5,e=i,i=""):"/"===n&&(_<5||">"===t[f][l+1])?(u(),3===_&&(r=r[0]),_=r,(r=r[0]).push(2,0,_),_=0):" "===n||"\t"===n||"\n"===n||"\r"===n?(u(),_=2):i+=n),3===_&&"!--"===i&&(_=4,r=r[0])}return u(),r}(t)),n),arguments,[])).length>1?n:n[0]}var un=rn.bind(L);export{I as Component,j as Fragment,h as Signal,_ as batch,ht as cloneElement,m as computed,at as createContext,L as createElement,R as createRef,w as effect,L as h,un as html,ct as hydrate,U as isValidElement,C as options,st as render,a as signal,Y as toChildArray,u as untracked,Ft as useCallback,nn as useComputed,Mt as useContext,Wt as useDebugValue,$t as useEffect,Lt as useErrorBoundary,Ot as useId,Vt as useImperativeHandle,Dt as useLayoutEffect,At as useMemo,Nt as useReducer,Tt as useRef,tn as useSignal,en as useSignalEffect,Pt as useState};